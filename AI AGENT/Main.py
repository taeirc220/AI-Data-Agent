# We import our Manager to handle the logic
from Manager import ManagerAgent
from Data_Agent import DataAgent # הוספנו ייבוא של סוכן הנתונים לכאן

def start_app():
    """
    This function starts the interactive application loop.
    """
    print("=========================================")
    print("🤖 Welcome to the AI Data Department 🤖")
    print("=========================================")
    
    # --- טעינת הנתונים פעם אחת בלבד! ---
    print("⏳ Loading data into memory, please wait...")
    d_agent = DataAgent("online_retail_small.csv")
    df = d_agent.get_data()
    
    if df is None:
        print("❌ CRITICAL ERROR: Could not load data. Shutting down.")
        return
        
    print(f"✅ Data loaded successfully! Found {len(df)} rows.")
    print("Type 'exit' or 'quit' to close the app.\n")

    # במקום להעביר למנהל את שם הקובץ, אנחנו מעבירים לו את הנתונים המוכנים (df)
    manager = ManagerAgent(df)

    # The infinite loop that keeps the program running waiting for user input
    while True:
        # Prompt the user to type something
        user_input = input("\n👤 You: ").strip()

        # Check if the user wants to exit
        if user_input.lower() in ['exit', 'quit', 'יציאה']:
            print("👋 Shutting down the Data Department. Goodbye!")
            break
        
        # Prevent empty inputs
        if not user_input:
            continue

        # Pass the input to the Manager and get the response
        response = manager.handle_request(user_input)
        
        # Print the Manager's response
        print(f"\n👔 {response}")

# --- Application entry point ---
if __name__ == "__main__":
    start_app()