"""A few built-in chat / code / math conversations.

Used to build a temporary vocab mapping and the overfit batch until BP-02's streams exist.
Format matches Aurora's ``preprocess_conversations``: a list of ``{"role", "content"}`` turns.
"""

SAMPLE_CONVERSATIONS = [
    [
        {"role": "user", "content": "Explain speculative decoding in two sentences."},
        {
            "role": "assistant",
            "content": "Speculative decoding lets a small draft model guess several next tokens, which the "
            "large target model then checks in a single forward pass. Accepted guesses are kept and the "
            "first wrong one is replaced, so the output matches the target exactly but arrives faster.",
        },
    ],
    [
        {"role": "user", "content": "What are three tips for staying focused while studying?"},
        {
            "role": "assistant",
            "content": "1. Work in short blocks, such as 25 minutes, followed by a 5-minute break.\n"
            "2. Put your phone in another room and close unrelated browser tabs.\n"
            "3. Start each session with one clear goal, like finishing a problem set.",
        },
    ],
    [
        {"role": "user", "content": "Write a Python function that checks whether a string is a palindrome."},
        {
            "role": "assistant",
            "content": "```python\ndef is_palindrome(s: str) -> bool:\n"
            '    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())\n'
            "    return cleaned == cleaned[::-1]\n```\n"
            'For example, `is_palindrome("Racecar")` returns `True`.',
        },
    ],
    [
        {"role": "user", "content": "How do I read a CSV file into a list of dictionaries in Python?"},
        {
            "role": "assistant",
            "content": "Use the `csv` module's `DictReader`:\n```python\nimport csv\n\n"
            'with open("data.csv", newline="") as f:\n    rows = list(csv.DictReader(f))\n```\n'
            "Each row is a dictionary keyed by the header names.",
        },
    ],
    [
        {"role": "user", "content": "A train travels 120 km in 1.5 hours. What is its average speed in km/h?"},
        {
            "role": "assistant",
            "content": "Average speed = distance / time = 120 km / 1.5 h = 80 km/h.\n"
            "The train's average speed is 80 km/h.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Natalia sold clips to 48 friends in April and half as many in May. "
            "How many clips did she sell in total?",
        },
        {
            "role": "assistant",
            "content": "In May she sold 48 / 2 = 24 clips.\nIn total she sold 48 + 24 = 72 clips.\n#### 72",
        },
    ],
]
