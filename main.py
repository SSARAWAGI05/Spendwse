#!/usr/bin/env python3
import os
import glob
import re
import threading
from datetime import datetime, timedelta

import cv2
import numpy as np
import face_recognition

# LangChain & Ollama for LLM-based processing (Group Expense)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

llm = Ollama(
    model="mistral",
    temperature=0.9,   # controls randomness
    top_p=0.9          # controls nucleus sampling
)

# Hugging Face Transformers (shared by both modules)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------
# MongoDB Setup
# ---------------------------
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017")
db = client["expense_manager"]
groups_collection = db["groups"]
personal_budgets_collection = db["personal_budgets"]
personal_expenses_collection = db["personal_expenses"]
goals_collection = db["personal_goals"]

# =============================================================================
#                      MODEL LOADING (Background Thread)
# =============================================================================
model = None
tokenizer = None
model_loaded_event = threading.Event()

def load_model_background():
    """Loads the HF model and tokenizer in the background."""
    global model, tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained("SaiGaneshanM/classifier_fintech")
    tokenizer = AutoTokenizer.from_pretrained("SaiGaneshanM/classifier_fintech")
    model_loaded_event.set()  # Signal that the model is ready.

# Start background thread (daemon=True ensures it won't block program exit)
background_thread = threading.Thread(target=load_model_background, daemon=True)
background_thread.start()

# =============================================================================
#                     FACIAL RECOGNITION CODE
# =============================================================================
class SimpleFacerec:
    """Simple facial recognition class for detecting known faces."""
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25
        self.tolerance = 0.45
        self.margin_threshold = 0.03

    def load_encoding_images(self, images_path):
        images_files = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images_files)} encoding images found.")
        for img_path in images_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read {img_path}")
                continue
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(img_path)
            filename, _ = os.path.splitext(basename)
            encodings = face_recognition.face_encodings(rgb_img)
            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(filename.upper())
            else:
                print(f"Warning: No face found in {img_path}")
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None
            if best_match_index is not None and face_distances[best_match_index] < self.tolerance:
                sorted_indices = np.argsort(face_distances)
                if len(sorted_indices) > 1:
                    second_best = face_distances[sorted_indices[1]]
                    if (second_best - face_distances[best_match_index]) < self.margin_threshold:
                        name = "Uncertain"
                    else:
                        name = self.known_face_names[best_match_index]
                else:
                    name = self.known_face_names[best_match_index]
            else:
                name = "Unknown"
            face_names.append(name)
        # Scale face locations back to original frame size.
        face_locations = np.array(face_locations) / self.frame_resizing
        return face_locations.astype(int), face_names

# =============================================================================
#                       UTILITY FUNCTIONS
# =============================================================================
def overlay_text_input(window_name, prompt, base_frame):
    """
    Displays a text-input overlay on top of base_frame in the given window.
    Collects keystrokes until Enter is pressed.
    """
    input_str = ""
    while True:
        frame = base_frame.copy()
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (50, 50, 50), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, prompt + input_str, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)
        if key in [13, 10]:
            break
        elif key in [8, 127]:
            input_str = input_str[:-1]
        elif key != -1 and key < 128:
            input_str += chr(key)
    return input_str

# =============================================================================
#                    GROUP EXPENSE MANAGEMENT (MongoDB)
# =============================================================================

# ----- Group Functions -----
def create_group():
    group_name = input("Enter the name of the new group: ").strip()
    if groups_collection.find_one({"group_name": group_name}):
        print(f"Group '{group_name}' already exists.")
        return
    members = []
    while True:
        member = input("Enter member name (or press Enter to finish): ").strip()
        if not member:
            break
        members.append({"Name": member.upper(), "Balance": 0})
    if members:
        group_doc = {"group_name": group_name, "members": members, "transactions": []}
        groups_collection.insert_one(group_doc)
        print(f"Group '{group_name}' created with {len(members)} members.")
    else:
        print("No members added. Group not created.")

def delete_group():
    group_list = list(groups_collection.find({}, {"group_name": 1}))
    if not group_list:
        print("No groups exist.")
        return
    print("Existing groups:")
    for i, group in enumerate(group_list, 1):
        print(f"{i}. {group['group_name']}")
    try:
        choice = int(input("Enter the number of the group to delete: ")) - 1
        if 0 <= choice < len(group_list):
            group_to_delete = group_list[choice]['group_name']
            groups_collection.delete_one({"group_name": group_to_delete})
            print(f"Group '{group_to_delete}' has been deleted.")
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")

def view_groups():
    group_list = list(groups_collection.find({}, {"group_name": 1}))
    if not group_list:
        print("No groups exist.")
    else:
        print("Existing groups:")
        for i, group in enumerate(group_list, 1):
            print(f"{i}. {group['group_name']}")

def select_group():
    group_list = list(groups_collection.find({}, {"group_name": 1}))
    if not group_list:
        print("No groups exist. Please create a group first.")
        return None
    print("Select a group:")
    for i, group in enumerate(group_list, 1):
        print(f"{i}. {group['group_name']}")
    try:
        choice = int(input("Enter the number of the group: ")) - 1
        if 0 <= choice < len(group_list):
            return group_list[choice]['group_name']
        else:
            print("Invalid choice.")
            return None
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return None

def print_current_balances(group_name):
    group = groups_collection.find_one({"group_name": group_name})
    print(f"\nCurrent Balances for group '{group_name}':")
    for member in group['members']:
        print(f"{member['Name']}: {member['Balance']:.2f}")

def log_transaction(group_name, lenders, borrowers, remarks):
    transaction = {
        'Lenders': lenders,
        'Borrowers': borrowers,
        'Remarks': remarks,
        'Timestamp': datetime.now()
    }
    groups_collection.update_one({"group_name": group_name}, {"$push": {"transactions": transaction}})

def add_member_to_group(group_name, member_name):
    groups_collection.update_one(
        {"group_name": group_name},
        {"$push": {"members": {"Name": member_name.upper(), "Balance": 0}}}
    )
    print(f"{member_name.upper()} has been added to the group '{group_name}'.")

def remove_member_from_group(group_name, member_name):
    result = groups_collection.update_one(
        {"group_name": group_name},
        {"$pull": {"members": {"Name": member_name.upper()}}}
    )
    if result.modified_count:
        print(f"{member_name.upper()} has been removed from group '{group_name}'.")
    else:
        print(f"Member {member_name.upper()} not found in group '{group_name}'.")

def update_balance(group_name, name, amount):
    groups_collection.update_one(
        {"group_name": group_name, "members.Name": name.upper()},
        {"$inc": {"members.$.Balance": amount}}
    )

def create_group_split_response(lender, amount, members):
    split_amount = round(amount / (len(members) - 1), 2)
    response_lines = [
        f"Lender: {lender}",
        f"Amount Lent: {amount}"
    ]
    borrower_count = 1
    for member in members:
        if member != lender:
            response_lines.extend([
                f"Borrower {borrower_count}: {member}",
                f"Amount Borrowed {borrower_count}: {split_amount}"
            ])
            borrower_count += 1
    return "\n".join(response_lines)

def get_all_members(group_name):
    group = groups_collection.find_one({"group_name": group_name})
    return [member['Name'] for member in group['members']]

def is_member_in_group(group_name, member_name):
    return member_name.upper() in get_all_members(group_name)

def get_corrected_response(expense, error_description):
    correction_prompt = f'''
    CRITICAL: Generate ONLY the exact output format shown below. DO NOT include any explanatory text, descriptions, or additional information.
    DO NOT include phrases like "Output:", "Given your input:", or any other commentary.

    Original Transaction details: {expense}
    Error Description: {error_description}

    OUTPUT FORMAT RULES:
    1. Start directly with "Lender:" line
    2. Include only the specified lines
    3. No additional text before, between, or after the format

    FORMATS:

    1. One Lender, One Borrower:
    Lender: [name]
    Amount Lent: [amount]
    Borrower: [name]
    Amount Borrowed: [amount]

    2. One Lender, Multiple Borrowers (Equal Split):
    Lender: [name]
    Amount Lent: [amount]
    Borrower 1: [name]
    Amount Borrowed 1: [amount/n]
    Borrower 2: [name]
    Amount Borrowed 2: [amount/n]

    3. One Lender, Multiple Borrowers (Percentage Split):
    Lender: [name]
    Amount Lent: [amount]
    Borrower 1: [name]
    Amount Borrowed 1: [amount * percentage1]
    Borrower 2: [name]
    Amount Borrowed 2: [amount * percentage2]

    STRICT RULES:
    1. Start IMMEDIATELY with "Lender:" - no introductory text
    2. NO explanations or descriptions
    3. NO "Output:" or "Result:" headers
    4. NO comments about split calculations
    5. ONLY the exact format shown above
    6. Round all amounts to 2 decimal places
    '''
    return llm(correction_prompt)

def detect_group_split(expense_text):
    group_split_phrases = [
        "split among all", "split amongst all", "split between all",
        "for all members", "everyone", "split equally",
        "split among everyone", "split between everyone",
        "divide equally", "share equally"
    ]
    return any(phrase in expense_text.lower() for phrase in group_split_phrases)

def parse_transaction_lines(lines):
    lenders = []
    borrowers = []
    lender_pattern = re.compile(r'^Lender(?:\s*\d+)?\s*:\s*(.+)$', re.IGNORECASE)
    amount_lent_pattern = re.compile(r'^Amount\s*Lent(?:\s*\d+)?\s*:\s*(\d+\.?\d*)$', re.IGNORECASE)
    borrower_pattern = re.compile(r'^Borrower(?:\s*\d+)?\s*:\s*(.+)$', re.IGNORECASE)
    amount_borrowed_pattern = re.compile(r'^Amount\s*Borrowed(?:\s*\d+)?\s*:\s*(\d+\.?\d*)$', re.IGNORECASE)
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        lender_match = lender_pattern.match(line)
        if lender_match:
            lender_name = lender_match.group(1).strip().upper()
            i += 1
            if i < len(lines):
                amt_line = lines[i].strip()
                amt_match = amount_lent_pattern.match(amt_line)
                lender_amount = float(amt_match.group(1)) if amt_match else 0.0
            else:
                lender_amount = 0.0
            lenders.append({'Name': lender_name, 'Amount': lender_amount})
            i += 1
            continue

        borrower_match = borrower_pattern.match(line)
        if borrower_match:
            borrower_name = borrower_match.group(1).strip().upper()
            i += 1
            if i < len(lines):
                amt_line = lines[i].strip()
                amt_match = amount_borrowed_pattern.match(amt_line)
                borrower_amount = float(amt_match.group(1)) if amt_match else 0.0
            else:
                borrower_amount = 0.0
            borrowers.append({'Name': borrower_name, 'Amount': borrower_amount})
            i += 1
            continue
        i += 1

    return lenders, borrowers

def process_expense(group_name, expense_text=None, overlay_mode=False, window_name=None, base_frame=None):
    if not overlay_mode:
        if expense_text is None:
            expense = input("Enter your transaction details: ").lower().strip()
        else:
            expense = expense_text.lower().strip()
    else:
        if expense_text is None:
            expense = overlay_text_input(window_name, "Enter transaction details: ", base_frame).lower().strip()
        else:
            expense = expense_text.lower().strip()

    if detect_group_split(expense):
        prompt = f'''
        Given this transaction: {expense}
        Extract only the lender name and total amount.
        Respond in exactly this format:
        Lender: [name]
        Amount: [number]
        '''
    else:
        prompt = f'''
        CRITICAL: Generate ONLY the exact output format shown below. DO NOT include any explanatory text, descriptions, or additional information.
        DO NOT include phrases like "Output:", "Given your input:", or any other commentary.

        Transaction details: {expense}

        OUTPUT FORMAT RULES:
        1. Start directly with "Lender:" line
        2. Include only the specified lines
        3. No additional text before, between, or after the format

        FORMATS:

        1. One Lender, One Borrower:
        Lender: [name]
        Amount Lent: [amount]
        Borrower: [name]
        Amount Borrowed: [amount]

        2. One Lender, Multiple Borrowers (Equal Split):
        Lender: [name]
        Amount Lent: [amount]
        Borrower 1: [name]
        Amount Borrowed 1: [amount/n]
        Borrower 2: [name]
        Amount Borrowed 2: [amount/n]

        3. One Lender, Multiple Borrowers (Percentage Split):
        Lender: [name]
        Amount Lent: [amount]
        Borrower 1: [name]
        Amount Borrowed 1: [amount * percentage1]
        Borrower 2: [name]
        Amount Borrowed 2: [amount * percentage2]

        STRICT RULES:
        1. Start IMMEDIATELY with "Lender:" - no introductory text
        2. NO explanations or descriptions
        3. NO "Output:" or "Result:" headers
        4. NO comments about split calculations
        5. ONLY the exact format shown above
        6. Round all amounts to 2 decimal places
        '''
    response = llm(prompt)

    if not overlay_mode:
        while True:
            print("\nLLM Response:")
            print(response)
            user_confirmation = input("Is this response correct? (yes/no): ").lower().strip()
            if user_confirmation == 'yes':
                break
            else:
                print("Please describe what's wrong with the response:")
                error_description = input("What needs to be corrected? ")
                response = get_corrected_response(expense, error_description)
    else:
        while True:
            frame = base_frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (50,50,50), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            y0, dy = 30, 30
            for i, line in enumerate(response.split('\n')):
                y = y0 + i*dy
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, "Is this response correct? (y/n): ", (10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(0)
            if key in [ord('y'), ord('Y')]:
                break
            elif key in [ord('n'), ord('N')]:
                cv2.putText(frame, "Describe what's wrong: ", (10, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.imshow(window_name, frame)
                corrected_text = overlay_text_input(window_name, "Correction: ", base_frame)
                response = get_corrected_response(expense, corrected_text)
            else:
                continue

    lines = response.strip().split('\n')
    lenders = []
    borrowers = []
    non_members = set()
    i = 0
    while i < len(lines):
        if 'Lender' in lines[i]:
            lender = lines[i].split(':')[1].strip().upper()
            amount_val = float(lines[i+1].split(':')[1].strip())
            if not is_member_in_group(group_name, lender):
                non_members.add(lender)
            lenders.append({'Name': lender, 'Amount': amount_val})
            i += 2
        elif 'Borrower' in lines[i]:
            borrower = lines[i].split(':')[1].strip().upper()
            amount_val = float(lines[i+1].split(':')[1].strip())
            if not is_member_in_group(group_name, borrower):
                non_members.add(borrower)
            borrowers.append({'Name': borrower, 'Amount': amount_val})
            i += 2
        else:
            i += 1

    if non_members:
        if not overlay_mode:
            print("\nThe following people are not in the group:")
            for member in non_members:
                print(f"- {member}")
            add_members = input("Do you want to add these people to the group? (yes/no): ").lower().strip()
            if add_members == 'yes':
                for member in non_members:
                    add_member_to_group(group_name, member)
            else:
                print("Transaction cancelled as some members are not in the group.")
                return response
        else:
            frame = base_frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (50,50,50), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "Not in group: " + ", ".join(non_members), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, "Add them? (y/n): ", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(0)
            if key in [ord('y'), ord('Y')]:
                for member in non_members:
                    add_member_to_group(group_name, member)
            else:
                cv2.putText(frame, "Transaction cancelled.", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1000)
                return response

    for lender in lenders:
        update_balance(group_name, lender['Name'], lender['Amount'])
    for borrower in borrowers:
        update_balance(group_name, borrower['Name'], -borrower['Amount'])
    log_transaction(group_name, lenders, borrowers, expense)
    if overlay_mode:
        frame = base_frame.copy()
        cv2.putText(frame, "Balances updated and transaction logged.", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1000)
    else:
        print("\nBalances updated and transaction logged in MongoDB.")
    return response

def suggest_payments(group_name):
    group = groups_collection.find_one({"group_name": group_name})
    members = group['members']
    members_copy = [member.copy() for member in members]
    debtors = sorted([m for m in members_copy if m['Balance'] < 0], key=lambda x: x['Balance'])
    creditors = sorted([m for m in members_copy if m['Balance'] > 0], key=lambda x: x['Balance'], reverse=True)
    
    suggestions = []
    for debtor in debtors:
        while debtor['Balance'] < 0 and creditors:
            creditor = creditors[0]
            amount = min(-debtor['Balance'], creditor['Balance'])
            suggestions.append(f"{debtor['Name']} should pay {amount:.2f} to {creditor['Name']}")
            debtor['Balance'] += amount
            creditor['Balance'] -= amount
            if creditor['Balance'] == 0:
                creditors.pop(0)
    return suggestions

def clear_all_accounts(group_name):
    groups_collection.update_one(
        {"group_name": group_name},
        {"$set": {"members.$[].Balance": 0}}
    )
    print(f"All accounts in group '{group_name}' have been cleared. All balances are now 0.")

def check_balance_via_face(group_name):
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    window_name = "Facial Recognition Balance"
    cv2.namedWindow(window_name)
    plus_buttons = []
    current_frame = None

    def on_mouse_event(event, x, y, flags, param):
        nonlocal current_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            for button in plus_buttons:
                cx, cy = button['center']
                radius = button['radius']
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    frozen_frame = current_frame.copy()
                    expense_input = overlay_text_input(window_name, f"Enter transaction details for {button['name']}: ", frozen_frame)
                    process_expense(group_name, expense_text=expense_input, overlay_mode=True, window_name=window_name, base_frame=frozen_frame)
                    break

    cv2.setMouseCallback(window_name, on_mouse_event)

    print("Press 'Esc' to exit facial recognition mode.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        current_frame = frame.copy()
        plus_buttons.clear()
        group = groups_collection.find_one({"group_name": group_name})
        sfr_faces = sfr.detect_known_faces(frame)
        face_locations, face_names = sfr_faces
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if name not in ["Unknown", "Uncertain"]:
                member = next((m for m in group['members'] if m['Name'] == name), None)
                if member:
                    balance = member.get("Balance", 0)
                    label = f"{name}: {balance:.2f}"
                    color = (0, 0, 255) if balance < 0 else (0, 255, 0)
                else:
                    label = f"{name}: No record"
                    color = (255, 255, 255)
            else:
                label = name
                color = (255, 255, 255)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

            if name not in ["Unknown", "Uncertain"]:
                button_radius = 15
                button_center = (right - button_radius, bottom - button_radius)
                cv2.circle(frame, button_center, button_radius, (50, 205, 50), -1)
                cv2.line(frame, (button_center[0] - 7, button_center[1]),
                         (button_center[0] + 7, button_center[1]), (255, 255, 255), 2)
                cv2.line(frame, (button_center[0], button_center[1] - 7),
                         (button_center[0], button_center[1] + 7), (255, 255, 255), 2)
                plus_buttons.append({"name": name, "center": button_center, "radius": button_radius})

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def manage_group():
    group_name = select_group()
    if not group_name:
        return
    while True:
        print(f"\nManaging group: {group_name}")
        print("1. Add expense")
        print("2. View balances")
        print("3. Suggest payments")
        print("4. Clear all accounts")
        print("5. Check Balance via Facial Recognition")
        print("6. Add member to group")
        print("7. Remove member from group")
        print("8. Back to group menu")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            process_expense(group_name)
        elif choice == '2':
            print_current_balances(group_name)
        elif choice == '3':
            suggestions = suggest_payments(group_name)
            print("\nSuggested Payments:")
            for suggestion in suggestions:
                print(suggestion)
        elif choice == '4':
            clear_all_accounts(group_name)
        elif choice == '5':
            group_name = select_group()
            if group_name:
                print(f"Starting facial recognition for group: {group_name}")
                check_balance_via_face(group_name)
        elif choice == '6':
            new_member = input("Enter the name of the new member: ").strip()
            if new_member:
                add_member_to_group(group_name, new_member)
        elif choice == '7':
            member_to_remove = input("Enter the name of the member to remove: ").strip()
            if member_to_remove:
                remove_member_from_group(group_name, member_to_remove)
        elif choice == '8':
            break
        else:
            print("Invalid choice. Please try again.")

def group_main_menu():
    while True:
        print("\nGroup Expense Management")
        print("1. Create a group")
        print("2. Delete a group")
        print("3. View groups")
        print("4. Manage group")
        print("5. Back to main menu")

        choice = input("Enter your choice: ").strip()
        if choice == '1':
            create_group()
        elif choice == '2':
            delete_group()
        elif choice == '3':
            view_groups()
        elif choice == '4':
            manage_group()
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

# =============================================================================
#                        PERSONAL FINANCE MANAGEMENT (MongoDB)
# =============================================================================
def get_current_week_window():
    now = datetime.now()
    start_of_week = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)
    return start_of_week, end_of_week

def extract_amount(expense_text):
    amounts = re.findall(r'\d+(?:\.\d+)?', expense_text)
    return float(amounts[0]) if amounts else 0.0

def categorize_transaction(expense_text):
    if not model_loaded_event.is_set():
        print("Model is still loading, please wait a moment...")
        model_loaded_event.wait()
    inputs = tokenizer(expense_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs)
    predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    valid_categories = [
        "Food & Dining", "Groceries", "Rent & Housing", "Utilities",
        "Transportation", "Health", "Shopping", "Entertainment",
        "Travel", "Education", "Stationery", "Savings & Investments",
        "Gifts & Donations", "Household & Maintenance", "Miscellaneous"
    ]
    if predicted_label not in valid_categories:
        predicted_label = "Miscellaneous"
    return predicted_label

def add_expense():
    expense_text = input("\nEnter your expense (e.g., 'I spent 50 on food today'): ").strip()
    amount = extract_amount(expense_text)
    
    if amount == 0.0:
        print("Could not extract a valid amount from your input. Please try again.")
        return

    category = categorize_transaction(expense_text)
    
    expense_record = {
        "amount": amount,
        "category": category,
        "description": expense_text,
        "timestamp": datetime.now()
    }
    personal_expenses_collection.insert_one(expense_record)
    print(f"Recorded: {amount:.2f} spent in the '{category}' category.")
    
    budgets_doc = personal_budgets_collection.find_one({"_id": "budgets"})
    if budgets_doc and category in budgets_doc.get("categories", {}):
        weekly_budget = budgets_doc["categories"][category]
        start_of_week, end_of_week = get_current_week_window()
        total_spent = sum(exp["amount"] for exp in personal_expenses_collection.find({
            "category": category,
            "timestamp": {"$gte": start_of_week, "$lte": end_of_week}
        }))
        if total_spent >= weekly_budget:
            print(f"ALERT: You have exceeded your WEEKLY budget for {category}!")
            print(f"Spent: {total_spent:.2f} / Weekly Budget: {weekly_budget:.2f}")
        elif total_spent >= 0.8 * weekly_budget:
            print(f"WARNING: You are nearing your WEEKLY budget for {category}.")
            print(f"Spent: {total_spent:.2f} / Weekly Budget: {weekly_budget:.2f}")
    else:
        print("No budget set for this category.")

def view_expenses():
    print("\nThis Week's Expenses:")
    start_of_week, end_of_week = get_current_week_window()
    expenses = personal_expenses_collection.find({
        "timestamp": {"$gte": start_of_week, "$lte": end_of_week}
    })
    for expense in expenses:
        ts = expense['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts} | {expense['category']} | {expense['amount']:.2f} | {expense['description']}")

def view_budgets():
    budgets_doc = personal_budgets_collection.find_one({"_id": "budgets"})
    if not budgets_doc:
        print("No budgets set up.")
        return
    print("\nWeekly Budgets and Spending:")
    categories = budgets_doc.get("categories", {})
    start_of_week, end_of_week = get_current_week_window()
    for category, budget in categories.items():
        total_spent = sum(exp["amount"] for exp in personal_expenses_collection.find({
            "category": category,
            "timestamp": {"$gte": start_of_week, "$lte": end_of_week}
        }))
        print(f"{category}: Weekly Budget = {budget:.2f}, Spent = {total_spent:.2f}")

def set_budgets():
    print("\nPlease enter your WEEKLY budgets for the following key categories:")
    budgets = {}
    budgets["Food & Dining"] = float(input("Food & Dining: "))
    budgets["Travel"] = float(input("Travel: "))
    budgets["Entertainment"] = float(input("Entertainment: "))
    budgets["Shopping"] = float(input("Shopping: "))
    budgets["Education"] = float(input("Education: "))
    personal_budgets_collection.replace_one(
        {"_id": "budgets"},
        {"_id": "budgets", "categories": budgets},
        upsert=True
    )
    print("Weekly budgets have been set up.\n")

def initialize_budgets():
    if personal_budgets_collection.find_one({"_id": "budgets"}):
        print("Budgets already exist.")
        update = input("Do you want to update your budgets? (yes/no): ").lower().strip()
        if update == "yes":
            set_budgets()
    else:
        set_budgets()

def create_savings_goal():
    goal_name = input("Enter the name of your savings goal (e.g., 'iPad'): ").strip()
    goal_amount = float(input("Enter the total amount you want to save: ").strip())
    
    # Check if a goal with this name already exists
    existing_goal = goals_collection.find_one({"name": goal_name})
    if existing_goal:
        print(f"A goal named '{goal_name}' already exists.")
        update = input("Do you want to update the existing goal? (yes/no): ").lower().strip()
        if update != 'yes':
            return
    
    goal_doc = {
        "name": goal_name,
        "total_amount": goal_amount,
        "current_savings": 0,
        "start_date": datetime.now(),
        "active": True
    }
    
    goals_collection.replace_one(
        {"name": goal_name},
        goal_doc,
        upsert=True
    )
    print(f"Savings goal '{goal_name}' created. Target: {goal_amount:.2f}")

def track_weekly_savings():
    # Retrieve all active goals
    active_goals = list(goals_collection.find({"active": True}))
    
    if not active_goals:
        print("No active savings goals found.")
        return
    
    # Retrieve budget information
    budgets_doc = personal_budgets_collection.find_one({"_id": "budgets"})
    if not budgets_doc:
        print("No budgets set up. Please set up budgets first.")
        return
    
    start_of_week, end_of_week = get_current_week_window()
    
    for goal in active_goals:
        total_savings = 0
        savings_breakdown = {}
        
        # Check savings in each budget category
        for category, weekly_budget in budgets_doc.get("categories", {}).items():
            # Calculate total spent in this category this week
            total_spent = sum(exp["amount"] for exp in personal_expenses_collection.find({
                "category": category,
                "timestamp": {"$gte": start_of_week, "$lte": end_of_week}
            }))
            
            # Calculate savings
            category_savings = max(0, weekly_budget - total_spent)
            if category_savings > 0:
                total_savings += category_savings
                savings_breakdown[category] = category_savings
        
        # Update goal savings
        if total_savings > 0:
            goals_collection.update_one(
                {"name": goal["name"]},
                {
                    "$inc": {"current_savings": total_savings},
                    "$push": {
                        "savings_history": {
                            "date": datetime.now(),
                            "amount": total_savings,
                            "breakdown": savings_breakdown
                        }
                    }
                }
            )
            
            # Fetch updated goal
            updated_goal = goals_collection.find_one({"name": goal["name"]})
            
            print(f"\nSavings Goal: {goal['name']}")
            print(f"Weekly Savings: {total_savings:.2f}")
            print(f"Current Total Savings: {updated_goal['current_savings']:.2f}")
            print(f"Target Amount: {goal['total_amount']:.2f}")
            
            # Check if goal is reached
            if updated_goal['current_savings'] >= goal['total_amount']:
                print(f"Congratulations! You've reached your goal for {goal['name']}!")
                goals_collection.update_one(
                    {"name": goal["name"]},
                    {"$set": {"active": False, "completion_date": datetime.now()}}
                )

def view_savings_goals():
    active_goals = list(goals_collection.find({"active": True}))
    completed_goals = list(goals_collection.find({"active": False}))
    
    print("\nActive Savings Goals:")
    for goal in active_goals:
        progress_percentage = (goal['current_savings'] / goal['total_amount']) * 100
        print(f"- {goal['name']}: {goal['current_savings']:.2f} / {goal['total_amount']:.2f} ({progress_percentage:.2f}%)")
    
    print("\nCompleted Savings Goals:")
    for goal in completed_goals:
        print(f"- {goal['name']}: Completed on {goal.get('completion_date', 'Unknown Date')}")

def manage_savings_goals():
    while True:
        print("\nSavings Goals Management")
        print("1. Create a New Savings Goal")
        print("2. View Savings Goals")
        print("3. Track Weekly Savings")
        print("4. Back to Personal Finance Menu")
        
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            create_savings_goal()
        elif choice == "2":
            view_savings_goals()
        elif choice == "3":
            track_weekly_savings()
        elif choice == "4":
            break
        else:
            print("Invalid choice. Please try again.")

def personal_main_menu():
    while True:
        print("\nPersonal Finance Management")
        print("1. Set up/Update Weekly Budgets")
        print("2. Add Expense")
        print("3. View This Week's Expenses")
        print("4. View Weekly Budgets and Spending")
        print("5. Savings Goals Management")
        print("6. Back to main menu")
        
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            set_budgets()
        elif choice == "2":
            add_expense()
        elif choice == "3":
            view_expenses()
        elif choice == "4":
            view_budgets()
        elif choice == "5":
            manage_savings_goals()
        elif choice == "6":
            break
        else:
            print("Invalid choice. Please try again.")

# =============================================================================
#                         TOP-LEVEL MAIN MENU
# =============================================================================
def main_menu():
    while True:
        print("\nSelect Expense Type:")
        print("1. Group Expense / Group Management")
        print("2. Personal Expense / Finance Tracker")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            group_main_menu()
        elif choice == "2":
            personal_main_menu()
            initialize_budgets()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

# =============================================================================
#                            PROGRAM ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main_menu()