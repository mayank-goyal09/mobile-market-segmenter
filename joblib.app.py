import joblib
import pandas as pd
import numpy as np
import time
import sys

def load_system():
    print("⏳ Loading Mobile Segmenter System...")
    time.sleep(0.5)
    try:
        artifacts = joblib.load('mobile_segmenter_model.pkl')
        print("✅ System Ready!\n")
        return artifacts
    except FileNotFoundError:
        print("❌ Error: 'mobile_segmenter_model.pkl' not found!")
        print("   Run the notebook first to save the model.")
        sys.exit()

# ==========================================
# 🛡️ POWER FUNCTION 1: STRICT NUMBER INPUT
# ==========================================
def get_valid_number(prompt, min_val=None, max_val=None):
    """
    Forces user to enter a number. 
    Optional: Enforce min/max range (e.g., Age 18-100).
    """
    while True:
        user_input = input(prompt).strip()
        try:
            value = float(user_input)
            
            # Check constraints
            if min_val is not None and value < min_val:
                print(f"   ❌ Too low! Must be at least {min_val}.")
                continue
            if max_val is not None and value > max_val:
                print(f"   ❌ Too high! Must be under {max_val}.")
                continue
                
            return value
        except ValueError:
            print(f"   ❌ Invalid input! '{user_input}' is not a number.")

# ==========================================
# 🛡️ POWER FUNCTION 2: STRICT MENU CHOICE
# ==========================================
def get_strict_choice(prompt, valid_options_map):
    """
    Forces user to pick a key from the map.
    Rejects EVERYTHING else. No accidents.
    """
    while True:
        choice = input(prompt).strip()
        
        # STRICT CHECK: Is the input actually a valid key?
        if choice in valid_options_map:
            return valid_options_map[choice]
            
        print(f"   ❌ Invalid choice! Please enter one of: {', '.join(valid_options_map.keys())}")

def get_user_input():
    print("📝 Enter Customer Details (Strict Mode 🔒):")
    print("-" * 30)
    
    # ✅ STRICT NUMERIC INPUTS
    age = get_valid_number("   🎂 Age (10-100): ", min_val=10, max_val=100)
    app_time = get_valid_number("   📱 App Usage (0-1440 min/day): ", min_val=0, max_val=1440)
    data_usage = get_valid_number("   📊 Data Usage (MB/day): ", min_val=0)
    battery = get_valid_number("   🔋 Battery Drain (mAh/day): ", min_val=0)
    screen_time = get_valid_number("   📺 Screen On Time (0-24 hours): ", min_val=0, max_val=24)
    num_apps = get_valid_number("   📲 Number of Apps Installed: ", min_val=0)
    
    # ✅ STRICT MENUS (No more "accidental" defaults!)
    
    print("\n   🤖 Operating System:")
    print("      1. Android")
    print("      2. iOS")
    os_sys = get_strict_choice("      Choice (1/2): ", 
                               {"1": "Android", "2": "iOS"})
    
    print("\n   👥 Gender:")
    print("      1. Male")
    print("      2. Female")
    # If you type '3' here, it will now REJECT it! 🚫
    gender = get_strict_choice("      Choice (1/2): ", 
                               {"1": "Male", "2": "Female"})
    
    print("\n   🏷️  Device Brand:")
    print("      1. Apple")
    print("      2. Samsung")
    print("      3. Xiaomi")
    print("      4. Google")
    print("      5. OnePlus")
    
    # Map options strictly
    brand_map = {
        "1": "iPhone 12",
        "2": "Samsung Galaxy S21",
        "3": "Xiaomi Mi 11",
        "4": "Google Pixel 5",
        "5": "OnePlus 9"
    }
    device = get_strict_choice("      Choice (1-5): ", brand_map)
    
    print(f"\n   ✅ Confirmed: {gender} | {os_sys} | {device}")

    # Create DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'App Usage Time (min/day)': [app_time],
        'Data Usage (MB/day)': [data_usage],
        'Battery Drain (mAh/day)': [battery],
        'Screen On Time (hours/day)': [screen_time],
        'Number of Apps Installed': [num_apps],
        'Operating System': [os_sys],
        'Gender': [gender],
        'Device Model': [device]
    })
    
    return input_data

def predict_segment(artifacts, input_data):
    model = artifacts['model']
    scaler = artifacts['scaler']
    le = artifacts['le']
    model_columns = artifacts['model_columns']
    
    input_dummies = pd.get_dummies(input_data)
    input_dummies = input_dummies.reindex(columns=model_columns, fill_value=0)
    
    input_scaled = scaler.transform(input_dummies)
    prediction_idx = model.predict(input_scaled)[0]
    
    # Convert class label to string safely
    prediction_name = str(le.inverse_transform([prediction_idx])[0])
    
    proba = model.predict_proba(input_scaled).max()
    
    return prediction_name, proba

def ask_continue():
    """
    Strict function to ask if user wants to continue.
    Only accepts 'y' or 'n' (case-insensitive).
    """
    while True:
        choice = input("\nTry another customer? (y/n): ").strip().lower()
        
        if choice == 'y':
            return True
        elif choice == 'n':
            return False
        else:
            print(f"   ❌ Invalid choice! '{choice}' is not valid.")
            print("   ✅ Please enter 'y' for Yes or 'n' for No.")

if __name__ == "__main__":
    print("========================================")
    print("📱 MOBILE MARKET SEGMENTER v2.0 (PRO) 📱")
    print("========================================\n")
    
    system = load_system()
    
    while True:
        try:
            user_data = get_user_input()
            
            print("\n🔄 Analyzing Usage Patterns...")
            time.sleep(1.0)
            
            segment, confidence = predict_segment(system, user_data)
            
            print("\n" + "="*40)
            print(f"🎯 PREDICTED SEGMENT: CLASS {segment}")
            print(f"💪 CONFIDENCE: {confidence:.2%}")
            print("="*40)
            
            # ✅ USE THE STRICT FUNCTION
            if ask_continue():
                print("\n🔄 Starting over...\n")
                continue
            else:
                print("\n👋 Thanks for using Mobile Market Segmenter!")
                print("🚀 Goodbye!\n")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Unexpected Error: {e}")
            print("   Try again with valid inputs.")

                
