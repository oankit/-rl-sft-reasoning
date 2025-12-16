import argparse
import random
import csv


def generate_brook_hills_question():
    """Generates a randomized 'Brook Hills High School' student demographics problem."""
    # Generate random values within reasonable ranges
    total_students = random.randint(1000, 5000)
    # Ensure even number for clean halves
    if total_students % 2 != 0:
        total_students += 1
    
    age_threshold = random.randint(14, 18)
    
    # Common fractions for proportions
    fractions = [
        (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)
    ]
    
    fraction1 = random.choice(fractions)
    fraction2 = random.choice(fractions)
    
    # Convert fractions to text
    fraction1_text = f"{fraction1[0]}/{fraction1[1]}" if fraction1[0] != 1 else f"one-{['', 'half', 'third', 'fourth', 'fifth'][fraction1[1]-1]}"
    fraction2_text = f"{fraction2[0]}/{fraction2[1]}" if fraction2[0] != 1 else f"one-{['', 'half', 'third', 'fourth', 'fifth'][fraction2[1]-1]}"
    
    # Algorithmic calculation
    over_age_students = total_students // 2
    under_age_students = total_students // 2
    
    over_age_males = int(over_age_students * (fraction1[0] / fraction1[1]))
    under_age_males = int(under_age_students * (fraction2[0] / fraction2[1]))
    
    total_males = over_age_males + under_age_males
    total_females = total_students - total_males
    
    # Generate question text
    question = f"Brook Hills High School currently enrolls {total_students:,} students. Half of these students are over {age_threshold} years old, and {fraction1_text} of the students over {age_threshold} years old are male. The remaining half of the students are under {age_threshold} years old, and {fraction2_text} of the students under {age_threshold} are male. In total, how many female students are enrolled at this school?.Please reason step by step, and put your final answer within \\boxed{{answer}} as an integer."
    
    return question, total_females


def generate_brownies_question():
    """Generates a randomized brownies calculation problem."""
    # Common names
    names = ["Quentin", "Michael", "George", "Fiona", "Yasmine", "Bob", "Alice", "Charlie"]
    name = random.choice(names)
    
    # Brownie types
    brownie_types = ["Nut Brownies", "Chewy Brownies", "Fudgy Brownies", "Cakey Brownies"]
    brownie_type = random.choice(brownie_types)
    
    # Generate values
    initial_dozen = random.randint(3, 12)
    
    # Office party brownies (fraction of a dozen)
    office_fractions = [(1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (1, 4), (1, 2), (3, 4)]
    of_frac = random.choice(office_fractions)
    office_amount_text = f"{of_frac[0]}/{of_frac[1]}"
    office_amount_val = of_frac[0] / of_frac[1]
    
    # Surprise party initial brownies (whole dozens)
    surprise_initial = random.randint(2, 10)
    
    # Eaten at party (mixed fraction: whole + fraction)
    eaten_whole = random.randint(1, initial_dozen + surprise_initial - 2)
    eaten_frac = random.choice(office_fractions)
    eaten_text = f"{eaten_whole} {eaten_frac[0]}/{eaten_frac[1]}"
    eaten_val = eaten_whole + (eaten_frac[0] / eaten_frac[1])
    
    # Calculate total remaining dozens then convert to individual count
    total_dozens = initial_dozen + office_amount_val + surprise_initial - eaten_val
    final_brownies = int(round(total_dozens * 12))
    
    question = f"{name} wanted brownies for her birthday. She made a batch for herself; {['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'][initial_dozen]} dozen {brownie_type} brownies. At her office, they threw her a party and sent her home with {office_amount_text} dozen brownies. When she arrived home, her friends were there to throw her a surprise party and had {surprise_initial} dozen brownies waiting. During the party, {eaten_text} dozen brownies were eaten. How many individual brownies did {name} have left over from the entire day?.Please reason step by step, and put your final answer within \\boxed{{answer}} as an integer."
    
    return question, final_brownies


def generate_homework_question():
    """Generates a randomized homework submission probability problem."""
    names = ["Yasmine", "Wendy", "Julia", "Uma", "Zach", "Diana", "Oscar", "Alice", "Bob", "Charlie"]
    name = random.choice(names)
    
    # Probabilities as integers (%)
    p_sub = random.choice(range(10, 80, 10))
    p_class_ext = random.choice(range(10, 80, 10))
    p_personal_ext = random.choice(range(5, 50, 5))
    
    # Homework is submitted ONLY if all conditions fail:
    # 1. No substitute (prob = 1 - p_sub)
    # 2. AND No class extension (prob = 1 - p_class_ext)
    # 3. AND No personal extension (prob = 1 - p_personal_ext)
    
    prob_no_sub = 1.0 - (p_sub / 100.0)
    prob_no_class_ext = 1.0 - (p_class_ext / 100.0)
    prob_no_personal_ext = 1.0 - (p_personal_ext / 100.0)
    
    final_prob_decimal = prob_no_sub * prob_no_class_ext * prob_no_personal_ext
    final_prob_percent = int(round(final_prob_decimal * 100))
    
    question = f"{name} is trying to decide whether they really need to do their homework. There's a {p_sub}% chance that tomorrow they'll have a substitute teacher who won't collect the homework. Even if the normal teacher comes in, there's a {p_class_ext}% chance she'll give everyone an extension. Even if the whole class doesn't get an extension, there's a {p_personal_ext}% chance {name} can convince the teacher their dog ate their assignment and get a personal extension. What is the percentage chance that {name} will actually have to turn in their homework tomorrow? Please reason step by step, and put your final answer within \\boxed{{answer}} as an integer."
    
    return question, final_prob_percent


def generate_truck_question():
    """Generates a randomized truck ordering cost calculation problem."""
    # Generate random values within reasonable ranges
    base_price = random.randint(25000, 50000)
    king_cab_price = random.randint(5000, 10000)
    
    # Calculate other prices based on relationships
    leather_seats_price = king_cab_price // 3  # one-third the cost of king cab
    running_boards_price = leather_seats_price - 500  # $500 less than leather seats
    light_package_price = random.randint(1000, 2000)  # upgraded exterior light package
    
    # Ensure running boards price is positive
    if running_boards_price <= 0:
        running_boards_price = 100
    
    # Algorithmic calculation
    total_cost = base_price + king_cab_price + leather_seats_price + running_boards_price + light_package_price
    
    # Generate question text
    question = f"Bill is ordering a new truck. He has decided to purchase a two-ton truck with several added features: a king cab upgrade, a towing package, leather seats, running boards, and the upgraded exterior light package. The base price of the truck is ${base_price:,}, and the other features are at extra cost. The king cab is an extra ${king_cab_price:,}, leather seats are one-third the cost of the king cab upgrade, running boards are $500 less than the leather seats, and the upgraded exterior light package is ${light_package_price:,}. What is the total cost of Bill's new truck, in dollars?.Please reason step by step, and put your final answer within \\boxed{{answer}} as an integer."
    
    return question, total_cost


def generate_newspaper_question():
    """Generates a randomized newspaper/tabloid page folding problem."""
    # Generate random values within reasonable ranges
    total_pages = random.randint(8, 64)  # Common tabloid sizes
    # Ensure page count is divisible by 4 for proper folding
    # Each sheet has 4 pages (left back, left front, right back, right front)
    if total_pages % 4 != 0:
        total_pages = (total_pages // 4 + 1) * 4
    
    # Algorithmic calculation: Each piece of paper has 4 pages (2 front, 2 back)
    # So number of pieces = total_pages / 4
    pieces_of_paper = total_pages // 4
    
    # Generate question text
    question = f"A simple folding newspaper or tabloid can be made by folding a piece of paper vertically and unfolding. Then, say, page 1 is printed on the left back, page 2 is printed on the left front, and then, perhaps page {total_pages} is printed on the right back, and page {total_pages-1} is printed on the right front. How many pieces of paper would be used in a {total_pages}-page tabloid?.Please reason step by step, and put your final answer within \\boxed{{answer}} as an integer."
    
    return question, pieces_of_paper


def generate_river_question():
    """Generates a randomized river water flow calculation problem."""
    # Generate random values within reasonable ranges
    initial_water = random.randint(2000, 8000)  # Initial water flow in gallons
    increase_amount = random.randint(3000, 10000)  # Additional increase on third day
    
    # Algorithmic calculation
    # Day 1: initial_water
    # Day 2: initial_water * 2 (doubles after heavy rain)
    # Day 3: (initial_water * 2) + increase_amount
    day2_water = initial_water * 2
    day3_water = day2_water + increase_amount
    
    # Generate question text
    question = f"The amount of water passing through a river at one point in time is {initial_water:,} gallons. After a day of heavy rain, the amount of water passing through the river doubles at the same point. If the volume of water passing through the river at that point increases by {increase_amount:,} gallons on the third day, calculate the total amount of water passing through the river at that point..Please reason step by step, and put your final answer within \\boxed{{answer}} as an integer."
    
    return question, day3_water


def generate_questions_csv(num_questions=1000, output_file="brook_hills_questions.csv", question_type="brook_hills"):
    """
    Generates a dataset of randomized math questions and saves to CSV.

    Args:
        num_questions: Total questions to generate.
        output_file: Path to output CSV.
        question_type: 'brook_hills', 'truck', 'newspaper', or 'river'.
    """
    questions_data = []
    
    for i in range(num_questions):
        if question_type == "922" or question_type == "brook_hills":
             question, answer = generate_brook_hills_question()
        elif question_type == "884" or question_type == "brownies":
             question, answer = generate_brownies_question()
        elif question_type == "584" or question_type == "truck":
            question, answer = generate_truck_question()
        elif question_type == "270" or question_type == "newspaper":
            question, answer = generate_newspaper_question()
        elif question_type == "785" or question_type == "river":
            question, answer = generate_river_question()
        elif question_type == "942" or question_type == "homework":
             question, answer = generate_homework_question()
        else:
            raise ValueError(f"question_type {question_type} not supported")
        
        questions_data.append([i, question, answer])
    
    # Write to CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'question', 'answer'])  # Header
        writer.writerows(questions_data)
    
    print(f"Generated {num_questions} {question_type} questions and saved to {output_file}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=1000)
    parser.add_argument("--output_file", type=str, default="questions.csv")
    parser.add_argument("--question_type", type=str, default="942")
    parser.add_argument("--all", action="store_true", default=False)
    args = parser.parse_args()
    
    if args.all:
        question_types = ["942", "922", "884", "584", "270", "785"]
        for question_type in question_types:
            generate_questions_csv(args.num_questions, args.output_file, question_type)
    else:
        generate_questions_csv(args.num_questions, args.output_file, args.question_type)
