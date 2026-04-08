from Data_Agent import DataAgent
from Manager import ManagerAgent
import os

def start_app():
    """
    מפעילה את אפליקציית סוכן ה-AI לניתוח נתוני ה-E-commerce.
    """
    print("=========================================")
    print("🤖 AI E-commerce Data Department 🤖")
    print("=========================================")
    
    # 1. טעינת הנתונים (Data Agent)
    file_name = "online_retail_small.csv"
    print(f"⏳ Loading data from {file_name}...")
    
    d_agent = DataAgent(file_name)
    df = d_agent.get_data()
    
    if df is None:
        print("❌ CRITICAL ERROR: Could not load data. Please check the file path.")
        return
        
    print(f"✅ Data loaded! Found {len(df)} rows.")
    
    # 2. אתחול ה-ManagerAgent המשודרג
    # כעת המנהל הוא סוכן אוטונומי מבוסס LangGraph
    print("🧠 Initializing AI Manager Agent...")
    manager = ManagerAgent(df)
    
    print("\nSystem is ready! Type 'exit' to quit.")
    print("Examples: 'Who is my top customer?', 'How are sales trending?', 'Hi, what can you do?'")

    # 3. לולאת השיחה (The Main Loop)
    while True:
        try:
            # קבלת קלט מהמשתמש
            user_input = input("\n👤 You: ").strip()

            # יציאה
            if user_input.lower() in ['exit', 'quit', 'יציאה', 'ביי']:
                print("👋 Goodbye! Shutting down the Data Department.")
                break
            
            if not user_input:
                continue

            # שליחה למנהל וקבלת תשובה (הסוכן יחליט אם להפעיל כלים או לענות חופשי)
            response = manager.handle_request(user_input)
            
            # הדפסת התשובה
            print(f"\n👔 Manager: {response}")

        except KeyboardInterrupt:
            print("\n👋 System interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    # וודא שיש לך מפתח API מוגדר בסביבה או בקובץ .env
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not found in environment variables!")
    
    start_app()