import asyncio
import csv
import calendar
import os
from datetime import date
from playwright.async_api import async_playwright

# Configuration
TARGET_TYPE = "Market Wise"
TARGET_COMMODITY_GROUP = "Vegetables"
TARGET_COMMODITY = "Onion"
TARGET_ARRIVAL_PRICE = "Both Arrival Quantity & Weighted Average Modal prices"
TARGET_PERIOD = "Date Wise"

# Date range for monthly scraping
START_YEAR = 2025
START_MONTH = 9
END_YEAR = 2025
END_MONTH = 12

OUTPUT_FILE = "agmarknet_onion_data.csv"

def get_monthly_ranges(start_year, start_month, end_year, end_month):
    ranges = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        start_date = date(y, m, 1)
        _, last_day = calendar.monthrange(y, m)
        end_date = date(y, m, last_day)
        ranges.append((start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return ranges

async def select_dropdown(page, label_text, option_text):
    print(f"Selecting '{option_text}' for '{label_text}'...")
    # Find the label that exactly matches the text, then get its parent container and the clickable peer
    import re
    label = page.locator("label").filter(has_text=re.compile(f"^{label_text}$")).first
    dropdown_clickable = label.locator("..").locator(".peer")
    
    # Force the click via javascript to bypass any overlapping elements like sticky headers
    await dropdown_clickable.evaluate("node => node.click()")
    
    # Wait for the dropdown options to populate/animate
    await page.wait_for_timeout(1000)
    
    option = page.get_by_text(option_text, exact=True).first
    await option.evaluate("node => node.click()")
    await page.wait_for_timeout(1000)

async def main():
    print("Starting Playwright...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            timezone_id='Asia/Kolkata',
            locale='en-GB'
        )
        page = await context.new_page()
        
        print('Navigating...')
        for attempt in range(3):
            try:
                await page.goto("https://agmarknet.gov.in/alltypeofreports", wait_until="domcontentloaded", timeout=600000)
                break
            except Exception as e:
                print(f"Navigation attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise e
                await page.wait_for_timeout(5000)
                
        await page.wait_for_timeout(2000)
        await page.wait_for_selector("text=All Type of Report")
        try:
            # 1. Select initial dropdowns
            await select_dropdown(page, "Type", TARGET_TYPE)
            await page.wait_for_timeout(2000) # Give it time to load States
            
            # Select State (Crucial to activate Submit button)
            await select_dropdown(page, "State", "All States")
            await page.wait_for_timeout(1000)

            await select_dropdown(page, "District", "All Districts")
            await page.wait_for_timeout(1000)

            await select_dropdown(page, "Market", "All Markets")
            await page.wait_for_timeout(1000)
            
            await select_dropdown(page, "Commodity Group", TARGET_COMMODITY_GROUP)
            await select_dropdown(page, "Commodity", TARGET_COMMODITY)
            await select_dropdown(page, "Arrival and Price", TARGET_ARRIVAL_PRICE)
            await select_dropdown(page, "Period", TARGET_PERIOD)
            # 2. Loop through months
            months_to_scrape = get_monthly_ranges(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
            
            # Write header if file doesn't exist or is empty
            header_written = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0

            for start_d, end_d in months_to_scrape:
                print(f"\nScraping data for period: {start_d} to {end_d}")
                
                # Set Dates
                await page.locator("#startDate").fill(start_d, force=True)
                await page.locator("#startDate").evaluate("node => node.dispatchEvent(new Event('change', { bubbles: true }))")
                
                await page.locator("#endDate").fill(end_d, force=True)
                await page.locator("#endDate").evaluate("node => node.dispatchEvent(new Event('change', { bubbles: true }))")
                await page.wait_for_timeout(500)

                submit_btn = page.locator('button', has_text="Submit").first
                await submit_btn.wait_for(state="visible", timeout=10000)
                
                # Check disabled attribute directly instead of waiting for pseudo-class which might glitch on React
                is_disabled = await submit_btn.get_attribute('disabled')
                while is_disabled is not None:
                     await page.wait_for_timeout(1000)
                     is_disabled = await submit_btn.get_attribute('disabled')
                
                # The submit button redirects to a new page, so we expect navigation
                print("Clicking Submit and expecting navigation...")
                try:
                    async with page.expect_navigation(timeout=600000):
                        await submit_btn.evaluate("node => node.click()")
                    
                    print(f"Navigated to results page: {page.url}")
                    
                    # Look for Export button
                    export_btn = page.locator('button', has_text="Export").first
                    await export_btn.wait_for(state="visible", timeout=30000)
                    
                    # Click Export to open the dropdown
                    await export_btn.click()
                    await page.wait_for_timeout(1000)
                    
                    # The dropdown contains list items like 'CSV' and 'Excel'
                    csv_option = page.get_by_text("Export as CSV").first
                    
                    # Expect the download
                    print("Initiating CSV download...")
                    async with page.expect_download(timeout=600000) as download_info:
                        await csv_option.click()                        
                    download = await download_info.value
                    
                    # Save temporarily
                    temp_path = f"temp_{start_d}_{end_d}.csv"
                    await download.save_as(temp_path)
                    print(f"Downloaded temporarily to {temp_path}")
                    
                    # Append data from downloaded CSV to our main CSV
                    with open(temp_path, 'r', encoding='utf-8') as temp_f:
                        reader = csv.reader(temp_f)
                        rows = list(reader)
                        
                        if len(rows) > 0:
                            header = rows[0]
                            data_rows = rows[1:]

                            with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as out_f:
                                writer = csv.writer(out_f)
                                if not header_written:
                                    writer.writerow(header)
                                    header_written = True
                                writer.writerows(data_rows)
                                
                            print(f"Saved {len(data_rows)} records to {OUTPUT_FILE}")
                        else:
                            print(f"No records found in downloaded CSV for {start_d} to {end_d}.")
                            
                    # Clean up temp file
                    os.remove(temp_path)

                except Exception as req_e:
                     print(f"Extraction failed for period {start_d} to {end_d}: {req_e}")                     
                
                # Navigate back to the form page for the next iteration
                print("Navigating back to form...")
                await page.goto("https://agmarknet.gov.in/alltypeofreports", wait_until="domcontentloaded", timeout=600000)
                await page.wait_for_timeout(3000)
                
                # Re-select dropdowns because we navigated away
                await select_dropdown(page, "Type", TARGET_TYPE)
                await page.wait_for_timeout(1000)
                await select_dropdown(page, "State", "All States")
                await page.wait_for_timeout(1000)
                await select_dropdown(page, "District", "All Districts")
                await page.wait_for_timeout(500)
                await select_dropdown(page, "Market", "All Markets")
                await page.wait_for_timeout(500)
                await select_dropdown(page, "Commodity Group", TARGET_COMMODITY_GROUP)
                await select_dropdown(page, "Commodity", TARGET_COMMODITY)
                await select_dropdown(page, "Arrival and Price", TARGET_ARRIVAL_PRICE)
                await select_dropdown(page, "Period", TARGET_PERIOD)
                

        except Exception as e:
            print(f"An error occurred during scraping: {e}")
        
        finally:
            print("Closing browser...")
            await browser.close()
if __name__ == "__main__":
    asyncio.run(main())

