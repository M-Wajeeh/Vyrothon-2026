"""
Synthetic data generation for on-device tool-calling fine-tuning.
Generates diverse training examples covering:
- All 5 tools with many prompt variations
- Multi-turn conversation with anaphora resolution
- Refusals (chitchat, unsupported tools, ambiguous with no history)
- Adversarial: misspellings, code-switching, unit ambiguity
- SHA-256 deduplication against public test set
"""
import json
import random
import os
import hashlib
import sys

# ============================================================
# TOOL CALL HELPER
# ============================================================
def tc(tool_name: str, args: dict) -> str:
    """Format a tool call response."""
    return f'<tool_call>{json.dumps({"tool": tool_name, "args": args})}</tool_call>'

# ============================================================
# ENTITY POOLS
# ============================================================
LOCATIONS = [
    "New York", "London", "Tokyo", "Paris", "Berlin", "Delhi", "Dubai",
    "Sydney", "Toronto", "Mumbai", "Shanghai", "Seoul", "Istanbul",
    "Cairo", "Lagos", "São Paulo", "Mexico City", "Bangkok", "Nairobi",
    "Jakarta", "Karachi", "Manila", "Rome", "Vienna", "Amsterdam",
    "Barcelona", "Lisbon", "Athens", "Prague", "Warsaw", "Dublin",
    "Helsinki", "Oslo", "Copenhagen", "Zurich", "Singapore", "Kuala Lumpur",
    "Riyadh", "Doha", "Muscat", "Colombo", "Dhaka", "Kathmandu",
    "Buenos Aires", "Lima", "Bogotá", "Santiago", "Casablanca",
    "Cape Town", "Accra", "Addis Ababa"
]

CURRENCIES_ISO = [
    "USD", "EUR", "GBP", "JPY", "INR", "AED", "CAD", "AUD", "CHF",
    "CNY", "KRW", "BRL", "MXN", "SGD", "HKD", "NZD", "SEK", "NOK",
    "DKK", "PLN", "THB", "MYR", "IDR", "PHP", "PKR", "BDT", "SAR",
    "QAR", "EGP", "NGN", "ZAR", "TRY", "RUB", "COP", "CLP", "PEN", "ARS"
]

DISTANCE_UNITS = ["meters", "feet", "kilometers", "miles", "inches", "cm", "yards", "millimeters"]
WEIGHT_UNITS = ["kilograms", "pounds", "grams", "ounces", "tons", "milligrams", "stones"]
TEMP_UNITS = ["celsius", "fahrenheit", "kelvin"]
VOLUME_UNITS = ["liters", "gallons", "milliliters", "cups", "pints", "quarts", "fluid ounces"]
SPEED_UNITS = ["km/h", "mph", "m/s", "knots"]
TIME_UNITS = ["seconds", "minutes", "hours", "days", "weeks"]
AREA_UNITS = ["square meters", "square feet", "acres", "hectares"]

ALL_UNIT_GROUPS = [DISTANCE_UNITS, WEIGHT_UNITS, TEMP_UNITS, VOLUME_UNITS, SPEED_UNITS, TIME_UNITS, AREA_UNITS]

DATES = [
    "2026-01-01", "2026-01-15", "2026-02-14", "2026-03-01", "2026-03-17",
    "2026-04-18", "2026-04-19", "2026-04-20", "2026-04-25", "2026-05-01",
    "2026-05-15", "2026-06-01", "2026-06-10", "2026-07-04", "2026-08-15",
    "2026-09-01", "2026-10-31", "2026-11-25", "2026-12-25", "2026-12-31",
]

TITLES = [
    "Doctor Appointment", "Team Meeting", "Gym Session", "Dinner Date",
    "Project Review", "Dentist Checkup", "Parent-Teacher Conference",
    "Birthday Party", "Code Review", "Sprint Planning", "Client Call",
    "Yoga Class", "Movie Night", "Book Club", "Grocery Shopping",
    "Car Service", "Flight Departure", "Interview Prep", "Haircut",
    "Lunch with Sarah", "Board Meeting", "Workshop", "Webinar",
    "Date Night", "Dog Walk", "Therapy Session", "Piano Lesson",
    "Soccer Practice", "Cooking Class", "Photography Session",
]

SQL_TABLES = ["users", "orders", "products", "employees", "customers", "transactions", "inventory", "logs"]
SQL_COLUMNS = {
    "users": ["id", "name", "email", "status", "signup_date", "age", "country"],
    "orders": ["id", "customer_id", "product_id", "amount", "order_date", "status"],
    "products": ["id", "name", "price", "category", "stock", "created_at"],
    "employees": ["id", "name", "department", "salary", "hire_date", "manager_id"],
    "customers": ["id", "name", "email", "phone", "city", "total_purchases"],
    "transactions": ["id", "user_id", "amount", "type", "timestamp", "status"],
    "inventory": ["id", "product_id", "quantity", "warehouse", "last_updated"],
    "logs": ["id", "level", "message", "timestamp", "source"],
}

# ============================================================
# WEATHER EXAMPLES
# ============================================================
def gen_weather():
    examples = []
    loc = random.choice(LOCATIONS)
    unit = random.choice(["C", "F"])
    
    templates_c = [
        f"What's the weather in {loc}?",
        f"How's the weather in {loc} today?",
        f"Tell me the current weather for {loc}.",
        f"Weather forecast for {loc}.",
        f"Is it cold or warm in {loc}?",
        f"What's the temperature in {loc}?",
        f"Check the weather in {loc} for me.",
        f"Can you look up the weather in {loc}?",
        f"I need to know the weather in {loc}.",
        f"Give me {loc} weather.",
    ]
    
    templates_f = [
        f"What's the weather in {loc} in Fahrenheit?",
        f"Tell me the temperature in {loc} in F.",
        f"Weather in {loc}, use Fahrenheit please.",
        f"How hot is it in {loc}? Give me Fahrenheit.",
    ]
    
    if unit == "F":
        prompt = random.choice(templates_f)
    else:
        prompt = random.choice(templates_c)
    
    examples.append({"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": tc("weather", {"location": loc, "unit": unit})}
    ]})
    return examples

# ============================================================
# CALENDAR EXAMPLES
# ============================================================
def gen_calendar():
    examples = []
    date = random.choice(DATES)
    title = random.choice(TITLES)
    
    action = random.choice(["create", "list"])
    
    if action == "create":
        templates = [
            f"Schedule a {title} on {date}.",
            f"Create an event called {title} for {date}.",
            f"Add {title} to my calendar for {date}.",
            f"Put a {title} on {date} in my calendar.",
            f"Book a {title} for {date}.",
            f"I need to schedule {title} on {date}.",
            f"Set up a {title} event on {date}.",
            f"Remind me about {title} on {date}.",
        ]
        prompt = random.choice(templates)
        examples.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": tc("calendar", {"action": "create", "date": date, "title": title})}
        ]})
    else:
        templates = [
            f"What's on my calendar for {date}?",
            f"Show my schedule for {date}.",
            f"Do I have anything planned on {date}?",
            f"List my events for {date}.",
            f"What am I doing on {date}?",
            f"Any meetings on {date}?",
            f"Check my calendar for {date}.",
            f"What appointments do I have on {date}?",
        ]
        prompt = random.choice(templates)
        examples.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": tc("calendar", {"action": "list", "date": date})}
        ]})
    
    return examples

# ============================================================
# CONVERT EXAMPLES
# ============================================================
def gen_convert():
    examples = []
    group = random.choice(ALL_UNIT_GROUPS)
    f_unit = random.choice(group)
    t_unit = random.choice([u for u in group if u != f_unit])
    val = round(random.uniform(0.1, 10000), random.choice([0, 1, 2]))
    if val == int(val):
        val = int(val)
    
    templates = [
        f"Convert {val} {f_unit} to {t_unit}.",
        f"How many {t_unit} are in {val} {f_unit}?",
        f"What is {val} {f_unit} in {t_unit}?",
        f"{val} {f_unit} = ? {t_unit}",
        f"Change {val} {f_unit} to {t_unit}.",
        f"How much is {val} {f_unit} in {t_unit}?",
        f"I need to convert {val} {f_unit} into {t_unit}.",
        f"What does {val} {f_unit} equal in {t_unit}?",
    ]
    prompt = random.choice(templates)
    
    examples.append({"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": tc("convert", {"value": val, "from_unit": f_unit, "to_unit": t_unit})}
    ]})
    return examples

# ============================================================
# CURRENCY EXAMPLES
# ============================================================
def gen_currency():
    examples = []
    f_curr = random.choice(CURRENCIES_ISO)
    t_curr = random.choice([c for c in CURRENCIES_ISO if c != f_curr])
    amt = random.choice([10, 25, 50, 75, 100, 150, 200, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 10000])
    
    templates = [
        f"Convert {amt} {f_curr} to {t_curr}.",
        f"How much is {amt} {f_curr} in {t_curr}?",
        f"Exchange {amt} {f_curr} to {t_curr}.",
        f"What's {amt} {f_curr} worth in {t_curr}?",
        f"Change {amt} {f_curr} into {t_curr}.",
        f"{amt} {f_curr} to {t_curr} please.",
        f"I have {amt} {f_curr}, how much is that in {t_curr}?",
        f"Give me the conversion for {amt} {f_curr} to {t_curr}.",
    ]
    prompt = random.choice(templates)
    
    examples.append({"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": tc("currency", {"amount": amt, "from": f_curr, "to": t_curr})}
    ]})
    return examples

# ============================================================
# SQL EXAMPLES
# ============================================================
SQL_PROMPT_QUERY_PAIRS = [
    ("Fetch names of employees earning more than 50000.", "SELECT name FROM employees WHERE salary > 50000"),
    ("Show all active users.", "SELECT * FROM users WHERE status = 'active'"),
    ("Count total orders placed this year.", "SELECT COUNT(*) FROM orders WHERE order_date >= '2026-01-01'"),
    ("Find the top 10 products by price.", "SELECT * FROM products ORDER BY price DESC LIMIT 10"),
    ("What's the average salary in each department?", "SELECT department, AVG(salary) FROM employees GROUP BY department"),
    ("List customers from New York.", "SELECT * FROM customers WHERE city = 'New York'"),
    ("Show me total revenue by month.", "SELECT MONTH(order_date) AS month, SUM(amount) AS revenue FROM orders GROUP BY MONTH(order_date)"),
    ("Get all transactions above 1000 dollars.", "SELECT * FROM transactions WHERE amount > 1000"),
    ("Who are the managers?", "SELECT DISTINCT manager_id FROM employees WHERE manager_id IS NOT NULL"),
    ("List products that are out of stock.", "SELECT * FROM products WHERE stock = 0"),
    ("Find users who joined in the last 30 days.", "SELECT * FROM users WHERE signup_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"),
    ("Show employees hired before 2025.", "SELECT * FROM employees WHERE hire_date < '2025-01-01'"),
    ("Get the total number of products in each category.", "SELECT category, COUNT(*) FROM products GROUP BY category"),
    ("List the most expensive product.", "SELECT * FROM products ORDER BY price DESC LIMIT 1"),
    ("Find orders with status pending.", "SELECT * FROM orders WHERE status = 'pending'"),
    ("Get all error logs from today.", "SELECT * FROM logs WHERE level = 'error' AND DATE(timestamp) = CURDATE()"),
    ("How many items are in warehouse A?", "SELECT SUM(quantity) FROM inventory WHERE warehouse = 'A'"),
    ("Delete inactive users.", "DELETE FROM users WHERE status = 'inactive'"),
    ("Update the price of product 42 to 29.99.", "UPDATE products SET price = 29.99 WHERE id = 42"),
    ("Show me the last 5 transactions.", "SELECT * FROM transactions ORDER BY timestamp DESC LIMIT 5"),
    ("Find customers who spent more than 10000 total.", "SELECT * FROM customers WHERE total_purchases > 10000"),
    ("What departments have more than 10 employees?", "SELECT department, COUNT(*) AS cnt FROM employees GROUP BY department HAVING cnt > 10"),
    ("Get the names and emails of all users.", "SELECT name, email FROM users"),
    ("Count how many orders each customer has.", "SELECT customer_id, COUNT(*) AS order_count FROM orders GROUP BY customer_id"),
    ("Show inventory items with less than 5 in stock.", "SELECT * FROM inventory WHERE quantity < 5"),
    ("Find the average order amount.", "SELECT AVG(amount) FROM orders"),
    ("List all unique product categories.", "SELECT DISTINCT category FROM products"),
    ("Who is the highest paid employee?", "SELECT * FROM employees ORDER BY salary DESC LIMIT 1"),
    ("Get the total number of users.", "SELECT COUNT(*) FROM users"),
    ("Show all completed transactions.", "SELECT * FROM transactions WHERE status = 'completed'"),
]

def gen_sql():
    examples = []
    prompt, query = random.choice(SQL_PROMPT_QUERY_PAIRS)
    
    examples.append({"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": tc("sql", {"query": query})}
    ]})
    return examples

# ============================================================
# REFUSAL EXAMPLES
# ============================================================
REFUSAL_CHITCHAT = [
    "Hello!", "Hi there!", "How are you?", "Good morning!", "What's up?",
    "Who made you?", "What are you?", "Tell me about yourself.",
    "Do you have feelings?", "Are you alive?",
    "Tell me a joke.", "Sing a song.", "Write me a poem.",
    "What do you think about AI?", "Do you dream?",
    "What's the meaning of life?", "What's your favorite color?",
    "Can you think?", "Are you conscious?", "Do you sleep?",
    "Thank you!", "Thanks!", "Great, bye!", "See you later!",
    "You're amazing!", "Good job!", "I love you.",
    "What year is it?", "Who is the president?", "Tell me a story.",
]

REFUSAL_UNSUPPORTED = [
    "Play some music.", "Set an alarm for 7 AM.", "Turn off the lights.",
    "Send an email to John.", "Call my mom.", "Text Mike about dinner.",
    "Order a pizza.", "Book a ride to the airport.", "Navigate to downtown.",
    "Take a screenshot.", "Open the camera.", "Record a voice memo.",
    "Search the web for recipes.", "Download this file.",
    "Post this on Twitter.", "Share this on Instagram.",
    "Translate this to French.", "Read my notifications.",
    "Change my wallpaper.", "Pair my Bluetooth headphones.",
    "Turn up the volume.", "Lock my phone.", "Restart the system.",
    "Find nearby restaurants.", "Book a hotel in Paris.",
    "Buy tickets for the concert.", "Track my package.",
    "Scan this QR code.", "Connect to WiFi.", "Open Netflix.",
]

REFUSAL_RESPONSES = [
    "I'm sorry, I can't do that. I can help you with weather lookups, calendar management, unit conversions, currency exchange, and SQL queries. How can I assist you with those?",
    "That's outside my capabilities. I'm designed to help with weather, calendar, unit conversions, currency conversions, and SQL queries. Want me to help with any of those?",
    "I don't have a tool for that. I can check weather, manage your calendar, convert units, exchange currencies, or run SQL queries. What would you like?",
    "I appreciate the request, but I can only help with weather, calendar events, unit conversions, currency exchange, and SQL queries.",
]

CHITCHAT_RESPONSES = [
    "Hello! I'm an AI assistant that can help you with weather, calendar, unit conversions, currency exchange, and SQL queries. How can I help?",
    "Hi! I'm here to assist you with weather lookups, calendar management, unit/currency conversions, and SQL queries. What do you need?",
    "I'm doing well! I can help you check weather, manage calendar events, convert units or currencies, and run SQL queries. What would you like to do?",
    "Thanks for asking! I'm a tool-calling assistant. I can help with weather, calendar, conversions, and SQL. How can I assist you today?",
]

def gen_refusal():
    examples = []
    
    if random.random() < 0.5:
        prompt = random.choice(REFUSAL_CHITCHAT)
        response = random.choice(CHITCHAT_RESPONSES)
    else:
        prompt = random.choice(REFUSAL_UNSUPPORTED)
        response = random.choice(REFUSAL_RESPONSES)
    
    examples.append({"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]})
    return examples

# ============================================================
# MULTI-TURN EXAMPLES
# ============================================================
def gen_multiturn():
    examples = []
    scenario = random.choice(["weather", "currency", "convert", "calendar"])
    
    if scenario == "weather":
        loc1 = random.choice(LOCATIONS)
        loc2 = random.choice([l for l in LOCATIONS if l != loc1])
        unit = random.choice(["C", "F"])
        followups = [f"And in {loc2}?", f"What about {loc2}?", f"How about {loc2}?", f"Now check {loc2}."]
        examples.append({"messages": [
            {"role": "user", "content": f"What's the weather in {loc1}?"},
            {"role": "assistant", "content": tc("weather", {"location": loc1, "unit": unit})},
            {"role": "user", "content": random.choice(followups)},
            {"role": "assistant", "content": tc("weather", {"location": loc2, "unit": unit})}
        ]})
    
    elif scenario == "currency":
        f_curr = random.choice(CURRENCIES_ISO)
        t_curr1 = random.choice([c for c in CURRENCIES_ISO if c != f_curr])
        t_curr2 = random.choice([c for c in CURRENCIES_ISO if c not in (f_curr, t_curr1)])
        amt = random.choice([50, 100, 200, 500, 1000])
        followups = [
            f"What about {t_curr2}?", f"And in {t_curr2}?",
            f"Now convert to {t_curr2} instead.", f"How much in {t_curr2}?",
        ]
        examples.append({"messages": [
            {"role": "user", "content": f"Convert {amt} {f_curr} to {t_curr1}."},
            {"role": "assistant", "content": tc("currency", {"amount": amt, "from": f_curr, "to": t_curr1})},
            {"role": "user", "content": random.choice(followups)},
            {"role": "assistant", "content": tc("currency", {"amount": amt, "from": f_curr, "to": t_curr2})}
        ]})
    
    elif scenario == "convert":
        group = random.choice(ALL_UNIT_GROUPS)
        if len(group) < 3:
            group = DISTANCE_UNITS
        f_unit = random.choice(group)
        t_unit1 = random.choice([u for u in group if u != f_unit])
        t_unit2 = random.choice([u for u in group if u not in (f_unit, t_unit1)])
        val = random.randint(1, 500)
        followups = [
            f"What about in {t_unit2}?", f"And in {t_unit2}?",
            f"Convert that to {t_unit2} too.", f"How much is that in {t_unit2}?",
        ]
        examples.append({"messages": [
            {"role": "user", "content": f"Convert {val} {f_unit} to {t_unit1}."},
            {"role": "assistant", "content": tc("convert", {"value": val, "from_unit": f_unit, "to_unit": t_unit1})},
            {"role": "user", "content": random.choice(followups)},
            {"role": "assistant", "content": tc("convert", {"value": val, "from_unit": f_unit, "to_unit": t_unit2})}
        ]})
    
    elif scenario == "calendar":
        date1 = random.choice(DATES)
        date2 = random.choice([d for d in DATES if d != date1])
        followups = [
            f"What about {date2}?", f"And on {date2}?",
            f"Check {date2} too.", f"Anything on {date2}?",
        ]
        examples.append({"messages": [
            {"role": "user", "content": f"What's on my calendar for {date1}?"},
            {"role": "assistant", "content": tc("calendar", {"action": "list", "date": date1})},
            {"role": "user", "content": random.choice(followups)},
            {"role": "assistant", "content": tc("calendar", {"action": "list", "date": date2})}
        ]})
    
    return examples

# ============================================================
# ADVERSARIAL / CODE-SWITCHING EXAMPLES
# ============================================================
ADVERSARIAL_WEATHER = [
    ("{loc} mein mausam kaisa hai?", "hi"),       # Hindi/Urdu
    ("{loc} ka moosam bata do", "hi"),             # Hindi
    ("Yeh batao {loc} mein garmi hai ya sardi?", "hi"),
    ("¿Cómo está el clima en {loc}?", "es"),       # Spanish
    ("Dime el clima de {loc}.", "es"),
    ("كيف حال الطقس في {loc}؟", "ar"),             # Arabic
    ("ما هو الطقس في {loc}؟", "ar"),
    ("Wethr in {loc}", "typo"),                     # Misspelling
    ("Wheather for {loc}", "typo"),
    ("Waht is the wether in {loc}?", "typo"),
    ("Check weathr {loc}", "typo"),
    ("temperture of {loc}", "typo"),
]

ADVERSARIAL_CURRENCY = [
    ("mujhe {amt} {f} ko {t} mein convert karo", "hi"),
    ("{amt} {f} kitne {t} hote hain?", "hi"),
    ("Convierte {amt} {f} a {t}.", "es"),
    ("¿Cuánto es {amt} {f} en {t}?", "es"),
    ("كم يساوي {amt} {f} بال{t}؟", "ar"),
    ("Convrrt {amt} {f} to {t}", "typo"),
    ("{amt} dollers to euroes", "typo"),
    ("Exchnge {amt} {f} into {t}", "typo"),
]

def gen_adversarial():
    examples = []
    
    if random.random() < 0.6:
        # Weather adversarial
        loc = random.choice(LOCATIONS)
        template, lang = random.choice(ADVERSARIAL_WEATHER)
        prompt = template.format(loc=loc)
        unit = "C"
        examples.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": tc("weather", {"location": loc, "unit": unit})}
        ]})
    else:
        # Currency adversarial
        f_curr = random.choice(CURRENCIES_ISO[:8])
        t_curr = random.choice([c for c in CURRENCIES_ISO[:8] if c != f_curr])
        amt = random.choice([50, 100, 200, 500, 1000])
        template, lang = random.choice(ADVERSARIAL_CURRENCY)
        prompt = template.format(amt=amt, f=f_curr, t=t_curr)
        examples.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": tc("currency", {"amount": amt, "from": f_curr, "to": t_curr})}
        ]})
    
    return examples

# ============================================================
# DEDUPLICATION
# ============================================================
def load_test_prompts(test_path: str) -> set:
    """Load and hash all prompts from the public test set."""
    hashes = set()
    if not os.path.exists(test_path):
        return hashes
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            for msg in ex.get("messages", []):
                if msg.get("role") == "user":
                    h = hashlib.sha256(msg["content"].strip().lower().encode()).hexdigest()
                    hashes.add(h)
    return hashes

def deduplicate(examples: list, test_hashes: set) -> list:
    """Remove examples whose user prompts match test set prompts."""
    clean = []
    removed = 0
    for ex in examples:
        dominated = False
        for msg in ex.get("messages", []):
            if msg.get("role") == "user":
                h = hashlib.sha256(msg["content"].strip().lower().encode()).hexdigest()
                if h in test_hashes:
                    dominated = True
                    break
        if not dominated:
            clean.append(ex)
        else:
            removed += 1
    if removed > 0:
        print(f"Removed {removed} examples that overlap with test set.")
    return clean

# ============================================================
# MAIN GENERATION
# ============================================================
def main():
    random.seed(42)
    
    all_examples = []
    
    # Generate diverse examples
    generators = [
        (gen_weather, 600),
        (gen_calendar, 400),
        (gen_convert, 500),
        (gen_currency, 500),
        (gen_sql, 400),
        (gen_refusal, 500),
        (gen_multiturn, 400),
        (gen_adversarial, 400),
    ]
    
    for gen_fn, count in generators:
        for _ in range(count):
            all_examples.extend(gen_fn())
    
    # Load teacher examples
    teacher_path = "starter/teacher_examples.jsonl"
    if os.path.exists(teacher_path):
        with open(teacher_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_examples.append(json.loads(line))
        print(f"Added teacher examples from {teacher_path}")
    
    # Deduplicate against public test set
    test_path = "starter/public_test.jsonl"
    test_hashes = load_test_prompts(test_path)
    all_examples = deduplicate(all_examples, test_hashes)
    
    # Deduplicate identical prompts within training set
    seen = set()
    unique = []
    for ex in all_examples:
        key = json.dumps(ex["messages"], sort_keys=True, ensure_ascii=False)
        h = hashlib.sha256(key.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    
    random.shuffle(unique)
    
    os.makedirs("data", exist_ok=True)
    train_file = "data/train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for ex in unique:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(unique)} unique training examples in {train_file}")
    
    # Print distribution
    tool_counts = {"weather": 0, "calendar": 0, "convert": 0, "currency": 0, "sql": 0, "refusal": 0, "multiturn": 0}
    for ex in unique:
        msgs = ex["messages"]
        assistant_msg = msgs[1]["content"] if len(msgs) > 1 else ""
        if len(msgs) > 3:
            tool_counts["multiturn"] += 1
        elif "<tool_call>" not in assistant_msg:
            tool_counts["refusal"] += 1
        else:
            try:
                call = json.loads(assistant_msg.split("<tool_call>")[1].split("</tool_call>")[0])
                tool_counts[call["tool"]] = tool_counts.get(call["tool"], 0) + 1
            except:
                pass
    print(f"Distribution: {json.dumps(tool_counts, indent=2)}")

if __name__ == "__main__":
    main()
