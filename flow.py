import pulp
import pandas as pd
import csv
import logging
from collections import defaultdict
import re
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def has_sgle(course_number):
    """Check if a course number indicates a seminar (ends in S/s)"""
    match = re.search(r'\d+[Ss]', course_number)
    return bool(match)

class CourseSelector:
    def __init__(self):
        self.AOK_CODES = {'ALP', 'CZ', 'NS', 'QS', 'SS'}
        self.MOI_CODES = {'CCI', 'STS', 'EI', 'R', 'W', 'FL'}
        self.ALL_CODES = self.AOK_CODES | self.MOI_CODES | {'SGLE'}
        
        # Requirements
        self.requirements = {
            code: 2 for code in self.AOK_CODES | (self.MOI_CODES - {'FL'})
        }
        self.requirements['FL'] = 3
        self.requirements['SGLE'] = 2  # SGLE capacity to sink

    def check_coverage(self, bins):
        """Debug function to check if requirements can potentially be met."""
        coverage = defaultdict(int)
        fl300_count = 0
        sgle_count = 0
        
        logging.info("\nAnalyzing code coverage:")
        for bin_key, bin_data in bins.items():
            bin_copies = min(bin_data['count'], 12)
            
            if bin_data['is_FL300']:
                fl300_count += bin_copies
                logging.info(f"FL300+ bin with codes {sorted(bin_key)}: {bin_copies} copies")
            
            for code in bin_key:
                coverage[code] += bin_copies
            
            if 'SGLE' in bin_key:
                sgle_count += bin_copies
        
        logging.info("\nPotential coverage per code:")
        insufficient = []
        for code, req_count in self.requirements.items():
            if coverage[code] < req_count:
                insufficient.append((code, coverage[code], req_count))
            logging.info(f"{code}: Available={coverage[code]}, Required={req_count}")
        
        if insufficient:
            logging.error("\nInsufficient coverage detected:")
            for code, available, required in insufficient:
                logging.error(f"{code}: Only {available} available but need {required}")
            return False
        
        logging.info(f"\nFL300+ courses available: {fl300_count}")
        logging.info(f"SGLE courses available: {sgle_count}")
        return True

    def parse_courses(self, filepath):
        """Parse course data from CSV file."""
        try:
            df = pd.read_csv(filepath, header=None, names=['raw'])
        except Exception as e:
            logging.error(f"Error reading file: {e}")
            return {}

        bins = defaultdict(lambda: {"count": 0, "is_FL300": False, "courses": []})
        
        for _, row in df.iterrows():
            parts = list(csv.reader([row['raw']]))[0]
            if len(parts) < 3:
                continue

            course_num, name, codes = parts[0].strip(), parts[1].strip(), parts[2].strip()
            codes = {code.strip() for code in codes.upper().split(',') if code.strip()}
            codes = codes & (self.AOK_CODES | self.MOI_CODES)  # Don't include SGLE from input

            # Add SGLE code if course number ends in S/s
            if has_sgle(course_num):
                codes.add('SGLE')

            if not codes:
                continue

            # Check if it's a 300+ level FL course
            is_fl300 = False
            if 'FL' in codes:
                course_level = re.search(r'\d+', course_num)
                if course_level and int(course_level.group()) >= 300:
                    is_fl300 = True

            bin_key = frozenset(codes)
            bins[bin_key]["count"] += 1
            bins[bin_key]["is_FL300"] = bins[bin_key]["is_FL300"] or is_fl300
            bins[bin_key]["courses"].append({
                "course_number": course_num,
                "name": name,
                "codes": sorted(codes)
            })

        return bins

    def debug_constraints(self, prob):
        """Debug function to analyze constraints."""
        logging.info("\nAnalyzing constraints:")
        for name, constraint in prob.constraints.items():
            rhs = constraint.constant
            lhs_terms = []
            for var, coef in constraint.items():
                var_name = var.name
                lhs_terms.append(f"{coef}*{var_name}")
            lhs = " + ".join(lhs_terms) if lhs_terms else "0"
            logging.info(f"{name}: {lhs} <= {rhs}")

    def analyze_infeasibility(self, prob, x, g1, g2, y_aok, y_moi, y_fl, y_sgle):
        """Analyze why the problem might be infeasible."""
        logging.error("\nInfeasibility Analysis:")
        
        # Check AOK coverage
        for code in self.AOK_CODES:
            max_possible = sum(1 for var in y_aok if code in var)
            logging.error(f"AOK {code}: Maximum possible={max_possible}, Required={self.requirements[code]}")
        
        # Check MOI coverage
        for code in self.MOI_CODES - {'FL'}:
            max_possible = sum(1 for var in y_moi if code in var)
            logging.error(f"MOI {code}: Maximum possible={max_possible}, Required={self.requirements[code]}")
        
        # Check FL coverage
        fl_max = sum(3 for var in y_fl)
        logging.error(f"FL: Maximum possible={fl_max}, Required={self.requirements['FL']}")
        
        # Check SGLE coverage
        sgle_max = len(y_sgle)
        logging.error(f"SGLE: Maximum possible={sgle_max}, Required={self.requirements['SGLE']}")
        
        # Check gadget distribution
        fl300_bins = sum(1 for var in g1)
        logging.error(f"FL300+ bins available: {fl300_bins}")

    def solve(self, bins):
        """Solve the course selection problem using ILP."""
        if not self.check_coverage(bins):
            logging.error("Problem is infeasible due to insufficient code coverage")
            return None

        prob = pulp.LpProblem("Course_Selection", pulp.LpMinimize)
        
        # Decision variables
        x = {}  # Bin selection
        g1 = {}  # Gadget 1 selection (MOI capacity 3, no FL)
        g2 = {}  # Gadget 2 selection (MOI capacity 2 + FL capacity 3)
        y_aok = {}  # AOK allocation
        y_moi = {}  # MOI allocation
        y_fl = {}   # FL allocation
        y_sgle = {} # SGLE allocation
        
        # Create variables
        logging.info("\nCreating decision variables...")
        for bin_key, bin_data in bins.items():
            num_copies = min(bin_data['count'], 12)
            for i in range(num_copies):
                bin_name = f"bin_{hash(bin_key)}_{i}"
                x[bin_name] = pulp.LpVariable(f"x_{bin_name}", cat='Binary')
                
                if bin_data['is_FL300']:
                    g1[bin_name] = pulp.LpVariable(f"g1_{bin_name}", cat='Binary')
                    g2[bin_name] = pulp.LpVariable(f"g2_{bin_name}", cat='Binary')
                
                # AOK allocations
                for code in bin_key & self.AOK_CODES:
                    y_aok[f"{bin_name}_{code}"] = pulp.LpVariable(
                        f"y_aok_{bin_name}_{code}", 
                        lowBound=0,
                        cat='Integer'
                    )
                
                # MOI allocations (excluding FL)
                for code in bin_key & (self.MOI_CODES - {'FL'}):
                    y_moi[f"{bin_name}_{code}"] = pulp.LpVariable(
                        f"y_moi_{bin_name}_{code}", 
                        lowBound=0,
                        cat='Integer'
                    )
                
                # FL allocations
                if 'FL' in bin_key:
                    y_fl[bin_name] = pulp.LpVariable(
                        f"y_fl_{bin_name}", 
                        lowBound=0,
                        cat='Integer'
                    )

                # SGLE allocations
                if 'SGLE' in bin_key:
                    y_sgle[bin_name] = pulp.LpVariable(
                        f"y_sgle_{bin_name}",
                        lowBound=0,
                        cat='Integer'
                    )

        # Objective: Minimize courses
        prob += pulp.lpSum(x[bin_name] for bin_name in x)

        logging.info("\nAdding constraints...")
        # Constraints
        constraint_count = 0
        for bin_key, bin_data in bins.items():
            num_copies = min(bin_data['count'], 12)
            for i in range(num_copies):
                bin_name = f"bin_{hash(bin_key)}_{i}"
                
                # SGLE constraint (up to 1 per course, independent of other flows)
                if 'SGLE' in bin_key:
                    prob += y_sgle[bin_name] <= 1 * x[bin_name], f"SGLE_Cap_{bin_name}"
                    constraint_count += 1
                
                # AOK capacity constraint (up to 1)
                aok_sum = pulp.lpSum(y_aok.get(f"{bin_name}_{code}", 0) 
                                   for code in bin_key & self.AOK_CODES)
                prob += aok_sum <= x[bin_name], f"AOK_Cap_{bin_name}"
                constraint_count += 1
                
                if bin_data['is_FL300']:
                    # FL300+ course constraints
                    # Only one gadget can be used
                    prob += g1[bin_name] + g2[bin_name] <= x[bin_name], f"Gadget_Select_{bin_name}"
                    constraint_count += 1
                    
                    # MOI allocation through gadgets
                    moi_sum = pulp.lpSum(y_moi.get(f"{bin_name}_{code}", 0) 
                                       for code in bin_key & (self.MOI_CODES - {'FL'}))
                    
                    # Gadget 1: Up to 3 MOI, no FL
                    prob += moi_sum <= 3 * g1[bin_name], f"G1_MOI_Cap_{bin_name}"
                    prob += y_fl[bin_name] <= 0 * g1[bin_name], f"G1_No_FL_{bin_name}"
                    constraint_count += 2
                    
                    # Gadget 2: Up to 2 MOI + up to 3 FL
                    prob += moi_sum <= 2 * g2[bin_name], f"G2_MOI_Cap_{bin_name}"
                    prob += y_fl[bin_name] <= 3 * g2[bin_name], f"G2_FL_Cap_{bin_name}"
                    constraint_count += 2
                    
                else:
                    # Regular course: up to 3 MOI codes total
                    moi_sum = pulp.lpSum(y_moi.get(f"{bin_name}_{code}", 0) 
                                       for code in bin_key & (self.MOI_CODES - {'FL'}))
                    fl_value = y_fl.get(bin_name, 0)
                    if fl_value or moi_sum:
                        prob += moi_sum + fl_value <= 3 * x[bin_name], f"MOI_Cap_{bin_name}"
                        constraint_count += 1

        # Meet minimum requirements
        logging.info("\nAdding requirement constraints...")
        
        # AOK requirements
        for code in self.AOK_CODES:
            prob += (
                pulp.lpSum(y_aok.get(f"{bin_name}_{code}", 0) for bin_name in x) >= 
                self.requirements[code], 
                f"Req_AOK_{code}"
            )
            constraint_count += 1
        
        # MOI requirements (excluding FL)
        for code in self.MOI_CODES - {'FL'}:
            prob += (
                pulp.lpSum(y_moi.get(f"{bin_name}_{code}", 0) for bin_name in x) >= 
                self.requirements[code], 
                f"Req_MOI_{code}"
            )
            constraint_count += 1
        
        # FL requirement
        prob += (
            pulp.lpSum(y_fl.get(bin_name, 0) for bin_name in x) >= 
            self.requirements['FL'], 
            "Req_FL"
        )
        constraint_count += 1

        # SGLE requirement
        prob += (
            pulp.lpSum(y_sgle.get(bin_name, 0) for bin_name in y_sgle) >= 
            1,  # Need at least 1 SGLE course
            "Req_SGLE"
        )
        constraint_count += 1

        logging.info(f"\nProblem size:")
        logging.info(f"Variables: {len(prob.variables())}")
        logging.info(f"Constraints: {constraint_count}")

        # Debug constraints before solving
        self.debug_constraints(prob)

        # Solve
        logging.info("\nSolving ILP...")
        prob.solve(pulp.PULP_CBC_CMD(msg=1))

        if pulp.LpStatus[prob.status] != 'Optimal':
            logging.error(f"\nNo solution found. Status: {pulp.LpStatus[prob.status]}")
            if pulp.LpStatus[prob.status] == 'Infeasible':
                logging.error("Analyzing infeasibility...")
                self.analyze_infeasibility(prob, x, g1, g2, y_aok, y_moi, y_fl, y_sgle)
            return None

        # Extract solution
        selected_courses = []
        logging.info("\nExtracting solution...")
        for bin_key, bin_data in bins.items():
            num_copies = min(bin_data['count'], 12)
            for i in range(num_copies):
                bin_name = f"bin_{hash(bin_key)}_{i}"
                if pulp.value(x[bin_name]) == 1:
                    course = bin_data["courses"].pop(0)
                    selected_courses.append(course)
                    
                    # Log allocations
                    allocations = []
                    
                    # Log AOK allocations
                    for code in bin_key & self.AOK_CODES:
                        if f"{bin_name}_{code}" in y_aok and pulp.value(y_aok[f"{bin_name}_{code}"]) > 0:
                            allocations.append(f"{code}={pulp.value(y_aok[f"{bin_name}_{code}"])}")
                    
                    # Log MOI allocations
                    for code in bin_key & (self.MOI_CODES - {'FL'}):
                        if f"{bin_name}_{code}" in y_moi and pulp.value(y_moi[f"{bin_name}_{code}"]) > 0:
                            allocations.append(f"{code}={pulp.value(y_moi[f"{bin_name}_{code}"])}")
                    
                    # Log FL allocations
                    if bin_name in y_fl and pulp.value(y_fl[bin_name]) > 0:
                        allocations.append(f"FL={pulp.value(y_fl[bin_name])}")

                    # Log SGLE allocations
                    if bin_name in y_sgle and pulp.value(y_sgle[bin_name]) > 0:
                        allocations.append(f"SGLE={pulp.value(y_sgle[bin_name])}")
                    
                    # Log which gadget was used for FL300+ courses
                    if bin_data['is_FL300']:
                        if g1[bin_name].value() == 1:
                            allocations.append("(Gadget 1)")
                        if g2[bin_name].value() == 1:
                            allocations.append("(Gadget 2)")
                    
                    logging.info(f"Course {course['course_number']}: {', '.join(allocations)}")

        return selected_courses

def main(filepath):
    selector = CourseSelector()
    bins = selector.parse_courses(filepath)
    
    if not bins:
        logging.error("No valid bins created from input data.")
        return
    
    logging.info("Created bins:")
    for bin_key, bin_data in bins.items():
        logging.info(f"Bin with codes {sorted(bin_key)}: {bin_data['count']} courses, FL300: {bin_data['is_FL300']}")
    
    selected_courses = selector.solve(bins)
    
    if selected_courses:
        logging.info("\nSelected courses:")
        for course in selected_courses:
            logging.info(f"{course['course_number']}: {course['name']} - Codes: {', '.join(course['codes'])}")
        
        with open('selected_courses.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Course Number', 'Name', 'Codes'])
            writer.writeheader()
            for course in selected_courses:
                writer.writerow({
                    'Course Number': course['course_number'],
                    'Name': course['name'],
                    'Codes': ', '.join(course['codes'])
                })
    else:
        logging.error("No valid solution found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <course_data.csv>")
    else:
        main(sys.argv[1])