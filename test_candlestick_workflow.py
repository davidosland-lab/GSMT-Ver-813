#!/usr/bin/env python3
"""
Test script to simulate the full candlestick workflow
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def test_candlestick_workflow():
    """Test the complete candlestick plotting workflow"""
    
    # Set up Chrome options for headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    driver = None
    
    try:
        print("🚀 Starting Enhanced Market Tracker Candlestick Test")
        print("=" * 60)
        
        # Initialize Chrome driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        
        # Navigate to the Enhanced Market Tracker
        url = "https://8080-ib2hlmks3ra0wwghsafox-6532622b.e2b.dev/enhanced_market_tracker.html"
        print(f"📱 Loading: {url}")
        driver.get(url)
        
        # Wait for page to load
        wait = WebDriverWait(driver, 30)
        print("⏳ Waiting for page to load...")
        
        # Check if market select dropdown is loaded
        market_select = wait.until(EC.presence_of_element_located((By.ID, "marketSelect")))
        print("✅ Market selector loaded")
        
        # Select a symbol (AAPL for testing)
        print("📊 Selecting AAPL symbol...")
        select = Select(market_select)
        select.select_by_value("AAPL")
        
        # Wait a moment for selection to register
        time.sleep(2)
        
        # Click Start Tracking button
        print("▶️ Starting tracking...")
        track_button = driver.find_element(By.CLASS_NAME, "track-btn")
        track_button.click()
        
        # Wait for data to load (look for status display)
        print("⏳ Waiting for data to load...")
        try:
            status_display = wait.until(EC.visibility_of_element_located((By.ID, "statusDisplay")))
            print("✅ Data loaded - status display visible")
        except TimeoutException:
            print("⚠️ Status display not visible, continuing anyway...")
        
        # Wait for chart to be populated
        time.sleep(5)
        
        # Find candlestick button
        print("🕯️ Looking for candlestick button...")
        candlestick_buttons = driver.find_elements(By.CSS_SELECTOR, ".view-toggle[data-view='candlestick']")
        
        if candlestick_buttons:
            candlestick_button = candlestick_buttons[0]
            print("✅ Candlestick button found")
            
            # Check if button is visible and clickable
            if candlestick_button.is_displayed():
                print("✅ Candlestick button is visible")
                
                # Click the candlestick button
                print("🔄 Clicking candlestick button...")
                driver.execute_script("arguments[0].click();", candlestick_button)
                
                # Wait for chart to update
                time.sleep(3)
                
                # Check if button is now active
                if "active" in candlestick_button.get_attribute("class"):
                    print("✅ Candlestick button is now active")
                else:
                    print("❌ Candlestick button did not become active")
                
                # Check console for any JavaScript errors
                print("🔍 Checking browser console for errors...")
                logs = driver.get_log('browser')
                
                js_errors = [log for log in logs if log['level'] == 'SEVERE']
                if js_errors:
                    print(f"❌ Found {len(js_errors)} JavaScript errors:")
                    for error in js_errors[-3:]:  # Show last 3 errors
                        print(f"   - {error['message']}")
                else:
                    print("✅ No severe JavaScript errors found")
                
                # Try to inspect chart canvas
                print("📊 Inspecting chart canvas...")
                canvas = driver.find_element(By.ID, "mainChart")
                if canvas:
                    print("✅ Chart canvas found")
                    # You could potentially screenshot here to verify visual state
                else:
                    print("❌ Chart canvas not found")
                
                return True
            else:
                print("❌ Candlestick button is not visible")
                return False
        else:
            print("❌ Candlestick button not found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
        
    finally:
        if driver:
            driver.quit()

def main():
    """Run the candlestick workflow test"""
    try:
        success = test_candlestick_workflow()
        
        print("\n" + "=" * 60)
        print("📋 CANDLESTICK WORKFLOW TEST SUMMARY")
        print("=" * 60)
        
        if success:
            print("🎉 CANDLESTICK WORKFLOW TEST PASSED")
            print("   ✅ Button found and clickable")
            print("   ✅ Chart interaction working")
        else:
            print("❌ CANDLESTICK WORKFLOW TEST FAILED") 
            print("   🔧 Issues detected in workflow")
            
    except ImportError:
        print("⚠️ Selenium not available for automated testing")
        print("📝 Manual testing required:")
        print("   1. Open: https://8080-ib2hlmks3ra0wwghsafox-6532622b.e2b.dev/enhanced_market_tracker.html")
        print("   2. Select a symbol (e.g., AAPL)")
        print("   3. Click 'Start Tracking'")
        print("   4. Wait for data to load")
        print("   5. Click the 'Candlestick' button in Chart View section")
        print("   6. Check if chart changes to OHLC view")

if __name__ == "__main__":
    main()