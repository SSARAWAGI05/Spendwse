# **SpendWise: AI-Powered Personal & Group Expense Manager**  

## **Problem Statement**  
Managing personal and group expenses is often tedious and error-prone, especially when dealing with:  
- **Large groups** where manual splitting becomes complicated  
- **Uneven expense distributions** (e.g., "Alice paid $100, but Bob only had a $20 meal")  
- **Lack of real-time insights** into spending habits  
- **Difficulty tracking budgets** and savings goals  
- **Manual reconciliation** of shared expenses  

**Current solutions (spreadsheets, basic apps) fail** because:  
❌ They require **manual calculations** for splits  
❌ They don’t handle **percentage-based or custom splits** easily  
❌ They lack **AI assistance** for natural language inputs  

**SpendWise solves this** with an **AI-driven expense manager** that:  
✔ **Automatically splits group expenses** (equal, percentage, or custom) via natural language  
✔ **Tracks personal budgets** with AI categorization  
✔ **Uses facial recognition** for instant balance checks  
✔ **Simplifies debt settlements** with smart suggestions  

---

## **Key Features**  

### **1. Smart Group Expense Management**  
- **AI-Powered Splitting** – Describe transactions naturally (e.g., *"John paid $100 for dinner, split 50-30-20 among 3 people"*) → AI handles the math.  
- **Facial Recognition Balance Check** – Use a webcam to instantly see who owes what.  
- **Custom Splits** – Supports **uneven splits, percentages, and partial payments**.  
- **Debt Simplification** – Recommends optimal repayments to settle group debts.  

### **2. Personal Finance Tracker**  
- **Auto-Categorization** – AI classifies expenses (e.g., *"$50 Uber ride"* → **Transportation**).  
- **Budget Alerts** – Warns before overspending.  
- **Savings Goals** – Tracks progress (e.g., *"Save $1,000 for a vacation"*).  

### **3. Facial Recognition Mode**  
- **Live balance checks** via webcam.  
- **One-click expense logging** on detected faces.  

---

## **System Requirements**  
- **Python 3.7+**  
- **MongoDB** (for data storage)  
- **Ollama** (for Mistral LLM)  
- **Webcam** (for facial recognition)  
- **CUDA-supported GPU (recommended)**  

---

## **Installation**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/SSARAWAGI05/SpendWise.git
cd SpendWise
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3. Set Up Ollama (For AI Expense Splitting)**  
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
```

### **4. Set Up MongoDB**  
- Install & run MongoDB locally.  
- The app auto-creates required collections (`groups`, `personal_budgets`, etc.).  

### **5. Add Face Encodings**  
- Place member photos in `images/` (e.g., `john.jpg`, `alice.png`).  

### **6. Run SpendWise**  
```bash
python spendwise.py
```

---

## **Usage Guide**  

### **1. Group Expenses**  
- **Create a group** → Add members.  
- **Log expenses** via natural language:  
  - *"Shubam paid Rs.120 for groceries, split equally among 4"*  
  - *"Kashvi paid Rs.200 for concert tickets, Shubam owes 40%, Aman 60%"*  
- **Check balances** via CLI or **facial recognition**.  

### **2. Personal Finance**  
- **Set budgets** (e.g., *"Rs.200/week on Food"*).  
- **Log expenses** → AI auto-categorizes them.  
- **Track savings goals**.  

### **3. Facial Recognition Mode**  
- Launch from group management.  
- Detects faces → shows balances.  
- **Click "+" on a face** to log expenses instantly.  

---

## **Why SpendWise?**  
✅ **No manual math** – AI handles complex splits.  
✅ **Works with large groups** – No spreadsheet headaches.  
✅ **Real-time tracking** – Always know who owes what.  

🚀 **Start managing money the smart way!** 🚀  

---

**GitHub**: [github.com/SSARAWAGI05/SpendWise](https://github.com/SSARAWAGI05/SpendWise)  
**Contact**: sarawagishubam@gmail.com
