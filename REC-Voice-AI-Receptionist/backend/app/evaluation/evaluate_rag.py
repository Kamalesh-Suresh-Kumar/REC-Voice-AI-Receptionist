from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

REPORT_DIR = PROJECT_ROOT / "evaluation_results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_REPORT = REPORT_DIR / "rag_evaluation_results.csv"
METRICS_REPORT = REPORT_DIR / "evaluation_metrics.csv"


# ============================================================
# IMPORT PROJECT SERVICES
# ============================================================

try:
    from app.rag.retriever import Retriever
    from app.rag.rag_service import RAGService

except Exception as exc:

    print()
    print("=" * 70)
    print("[IMPORT ERROR]")
    print(exc)
    print("=" * 70)
    print()
    print("Run this program from the backend directory:")
    print()
    print("python -m app.evaluation.evaluate_rag")
    print()

    sys.exit(1)


# ============================================================
# TEST DATASET
# ============================================================
#
# These questions are based on information already present
# in your currently ingested text documents:
#
#   - R2023 CSE Curriculum and Syllabus
#   - REC 2023 UG Regulations
#
# We intentionally DO NOT include the two scanned certificate
# PDFs here because your current ingestion output showed:
#
#   Pages with text: 0
#   Chunks: 0
#
# Those documents should be evaluated after OCR is implemented.
#
# expected_terms:
#     At least one of these terms should appear in the retrieved
#     chunks for the retrieval test.
#
# answer_terms:
#     ALL of these terms should appear in the generated answer
#     for the answer-correctness test.
#
# ============================================================

TEST_CASES = [{'id': 1,
  'category': 'Admissions',
  'question': 'Which undergraduate programme is offered in Computer Science and Engineering?',
  'expected_terms': ['B.E. COMPUTER SCIENCE AND ENGINEERING'],
  'answer_terms': ['B.E. COMPUTER SCIENCE AND ENGINEERING']},
 {'id': 2,
  'category': 'Admissions',
  'question': 'What postgraduate programme is offered in Computer Science and Engineering?',
  'expected_terms': ['M.E. COMPUTER SCIENCE AND ENGINEERING'],
  'answer_terms': ['M.E. COMPUTER SCIENCE AND ENGINEERING']},
 {'id': 3,
  'category': 'Admissions',
  'question': 'How long is the MBA programme listed by Rajalakshmi Engineering College?',
  'expected_terms': ['2 YEARS', 'MBA'],
  'answer_terms': ['2 YEARS']},
 {'id': 4,
  'category': 'Admissions',
  'question': 'Does Rajalakshmi Engineering College offer a Ph.D. programme in Computer Science and Engineering?',
  'expected_terms': ['PH.D.', 'COMPUTER SCIENCE AND ENGINEERING'],
  'answer_terms': ['PH.D.', 'COMPUTER SCIENCE AND ENGINEERING']},
 {'id': 5,
  'category': 'Admissions',
  'question': 'What is the opening cutoff for B.E. Computer Science and Engineering in the published cutoff table?',
  'expected_terms': ['198'],
  'answer_terms': ['198']},
 {'id': 6,
  'category': 'Admissions',
  'question': 'What is the closing cutoff for B.E. Computer Science and Engineering in the published cutoff table?',
  'expected_terms': ['192'],
  'answer_terms': ['192']},
 {'id': 7,
  'category': 'Admissions',
  'question': 'What is the opening cutoff for B.Tech. Artificial Intelligence and Data Science?',
  'expected_terms': ['197'],
  'answer_terms': ['197']},
 {'id': 8,
  'category': 'Admissions',
  'question': 'What is the closing cutoff for B.Tech. Information Technology?',
  'expected_terms': ['190'],
  'answer_terms': ['190']},
 {'id': 9,
  'category': 'Admissions',
  'question': 'What is the closing cutoff for B.E. Robotics and Automation?',
  'expected_terms': ['181'],
  'answer_terms': ['181']},
 {'id': 10,
  'category': 'Admissions',
  'question': 'What is the closing cutoff for B.Tech. Food Technology?',
  'expected_terms': ['173'],
  'answer_terms': ['173']},
 {'id': 11,
  'category': 'Academic programmes/courses',
  'question': 'What is the subject code for Database Management Systems?',
  'expected_terms': ['CS23332', 'DATABASE MANAGEMENT SYSTEMS'],
  'answer_terms': ['CS23332']},
 {'id': 12,
  'category': 'Academic programmes/courses',
  'question': 'What is the subject code for Computer Networks?',
  'expected_terms': ['CS23532', 'COMPUTER NETWORKS'],
  'answer_terms': ['CS23532']},
 {'id': 13,
  'category': 'Academic programmes/courses',
  'question': 'What concepts are covered in Database Management Systems?',
  'expected_terms': ['DATABASE', 'SQL', 'NORMALIZATION'],
  'answer_terms': ['DATABASE', 'SQL', 'NORMALIZATION']},
 {'id': 14,
  'category': 'Academic programmes/courses',
  'question': 'What is the duration of the B.E. or B.Tech. programme?',
  'expected_terms': ['8 SEMESTERS', 'FOUR ACADEMIC YEARS'],
  'answer_terms': ['8', 'SEMESTER']},
 {'id': 15,
  'category': 'Academic programmes/courses',
  'question': 'How many working days does each semester normally have?',
  'expected_terms': ['90 WORKING DAYS'],
  'answer_terms': ['90', 'WORKING DAYS']},
 {'id': 16,
  'category': 'Academic programmes/courses',
  'question': 'What is the prescribed range of total credits for a B.E./B.Tech. degree programme?',
  'expected_terms': ['160-165', '160', '165'],
  'answer_terms': ['160', '165']},
 {'id': 17,
  'category': 'Academic programmes/courses',
  'question': 'How many semesters is a B.E. or B.Tech. programme normally completed in?',
  'expected_terms': ['8 SEMESTERS'],
  'answer_terms': ['8', 'SEMESTER']},
 {'id': 18,
  'category': 'Academic programmes/courses',
  'question': 'What is the maximum duration for completing the B.E. or B.Tech. programme for HSC students?',
  'expected_terms': ['14 SEMESTERS'],
  'answer_terms': ['14', 'SEMESTER']},
 {'id': 19,
  'category': 'Academic programmes/courses',
  'question': 'What courses can a student register for from the second semester onwards?',
  'expected_terms': ['CURRENT SEMESTER', 'REAPPEARANCE'],
  'answer_terms': ['CURRENT', 'SEMESTER']},
 {'id': 20,
  'category': 'Academic programmes/courses',
  'question': 'When does course enrollment for semesters II to VIII commence?',
  'expected_terms': ['10 WORKING DAYS', 'PRECEDING SEMESTER'],
  'answer_terms': ['10', 'WORKING DAYS']},
 {'id': 21,
  'category': 'Academic programmes/courses',
  'question': 'What undergraduate programme is offered by the Information Technology department?',
  'expected_terms': ['B.TECH. INFORMATION TECHNOLOGY'],
  'answer_terms': ['INFORMATION TECHNOLOGY']},
 {'id': 22,
  'category': 'Academic programmes/courses',
  'question': 'What postgraduate programme is offered by the Information Technology department?',
  'expected_terms': ['M.TECH. DATA SCIENCE'],
  'answer_terms': ['M.TECH.', 'DATA SCIENCE']},
 {'id': 23,
  'category': 'Academic programmes/courses',
  'question': 'What programmes are offered by the Electronics and Communication Engineering department?',
  'expected_terms': ['B.E. ELECTRONICS AND COMMUNICATION ENGINEERING', 'M.E. COMMUNICATION SYSTEMS', 'PH.D.'],
  'answer_terms': ['ELECTRONICS AND COMMUNICATION ENGINEERING']},
 {'id': 24,
  'category': 'Academic programmes/courses',
  'question': 'What undergraduate programme is offered by the Artificial Intelligence and Machine Learning department?',
  'expected_terms': ['B.TECH. ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING'],
  'answer_terms': ['ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING']},
 {'id': 25,
  'category': 'Academic programmes/courses',
  'question': 'What minor programme in Data Science is listed by Rajalakshmi Engineering College?',
  'expected_terms': ['MINOR PROGRAMME IN DATA SCIENCE'],
  'answer_terms': ['DATA SCIENCE']},
 {'id': 26,
  'category': 'Departments/faculty',
  'question': 'Who is listed as Professor and Head of the Computer Science and Engineering department?',
  'expected_terms': ['P. KUMAR', 'HOD'],
  'answer_terms': ['KUMAR']},
 {'id': 27,
  'category': 'Departments/faculty',
  'question': 'Who is listed as Professor and Head of the Electronics and Communication Engineering department?',
  'expected_terms': ['M. PALANIVELAN', 'HOD'],
  'answer_terms': ['PALANIVELAN']},
 {'id': 28,
  'category': 'Departments/faculty',
  'question': 'Who is listed as Professor and Head of the Information Technology department?',
  'expected_terms': ['PRIYA VIJAY', 'HOD'],
  'answer_terms': ['PRIYA VIJAY']},
 {'id': 29,
  'category': 'Departments/faculty',
  'question': 'Who is listed as Professor and Head of the Artificial Intelligence and Machine Learning department?',
  'expected_terms': ['S. BAGHAVATHI PRIYA', 'HOD'],
  'answer_terms': ['BAGHAVATHI PRIYA']},
 {'id': 30,
  'category': 'Departments/faculty',
  'question': 'Who is listed as Professor and Head of the Computer Science and Design department?',
  'expected_terms': ['P. REVATHY', 'HOD'],
  'answer_terms': ['REVATHY']},
 {'id': 31,
  'category': 'Departments/faculty',
  'question': 'Who is listed as Professor and Head of the Mechanical Engineering department?',
  'expected_terms': ['S.P. SRINIVASAN', 'HOD'],
  'answer_terms': ['SRINIVASAN']},
 {'id': 32,
  'category': 'Departments/faculty',
  'question': 'Who is listed as Professor and Head of the Robotics and Automation department?',
  'expected_terms': ['R. GIRI', 'HOD'],
  'answer_terms': ['GIRI']},
 {'id': 33,
  'category': 'Departments/faculty',
  'question': 'Who is listed as Professor and Head of Management Studies?',
  'expected_terms': ['M. UMA RAMAN', 'HOD'],
  'answer_terms': ['UMA RAMAN']},
 {'id': 34,
  'category': 'Departments/faculty',
  'question': 'Who is listed as Professor and Head of Humanities and Sciences?',
  'expected_terms': ['M. MUTHUSAMY', 'HOD'],
  'answer_terms': ['MUTHUSAMY']},
 {'id': 35,
  'category': 'Departments/faculty',
  'question': 'Who is the Controller of Examinations listed in the Academic Council?',
  'expected_terms': ['R. KARTHIKEYAN', 'CONTROLLER OF EXAMINATIONS'],
  'answer_terms': ['KARTHIKEYAN']},
 {'id': 36,
  'category': 'Regulations/attendance',
  'question': 'What is the minimum attendance required for a course?',
  'expected_terms': ['75%', '75'],
  'answer_terms': ['75%']},
 {'id': 37,
  'category': 'Regulations/attendance',
  'question': 'What happens when attendance is between 65% and 74%?',
  'expected_terms': ['65%', '74%', 'MEDICAL', 'SPORTS'],
  'answer_terms': ['65%', '74%']},
 {'id': 38,
  'category': 'Regulations/attendance',
  'question': 'How many semesters can attendance concession be availed?',
  'expected_terms': ['TWO SEMESTERS'],
  'answer_terms': ['TWO']},
 {'id': 39,
 'category': 'Regulations/attendance',
 'question': 'What happens if a student does not satisfy the attendance requirements?',
 'expected_terms': [
     'SA',
     'SHORTAGE OF ATTENDANCE',
     'END SEMESTER EXAMINATIONS',
     'PREVENTED FROM WRITING'
 ],
 'answer_terms': [
     'SA',
     'SHORTAGE OF ATTENDANCE',
     'END SEMESTER EXAMINATIONS',
     'PREVENTED FROM WRITING'
 ]},
 {'id': 40,
  'category': 'Regulations/attendance',
  'question': 'What is the maximum number of courses that can be added or dropped in a semester?',
  'expected_terms': ['2 COURSES', 'ADD OR DROP'],
  'answer_terms': ['2', 'COURSES']},
 {'id': 41,
  'category': 'Regulations/attendance',
  'question': 'What is the medium of instruction at Rajalakshmi Engineering College?',
  'expected_terms': ['ENGLISH'],
  'answer_terms': ['ENGLISH']},
 {'id': 42,
  'category': 'Regulations/attendance',
  'question': 'What is the normal duration of the B.E./B.Tech. programme in semesters?',
  'expected_terms': ['8 SEMESTERS'],
  'answer_terms': ['8', 'SEMESTER']},
 {'id': 43,
  'category': 'Regulations/attendance',
  'question': 'What is the maximum programme duration allowed for HSC or equivalent students?',
  'expected_terms': ['14 SEMESTERS'],
  'answer_terms': ['14', 'SEMESTER']},
 {'id': 44,
  'category': 'Regulations/attendance',
  'question': 'How many working days are normally provided in a semester?',
  'expected_terms': ['90 WORKING DAYS'],
  'answer_terms': ['90', 'WORKING DAYS']},
 {'id': 45,
  'category': 'Regulations/attendance',
  'question': 'How many courses can be added or dropped at maximum in one semester?',
  'expected_terms': ['2 COURSES'],
  'answer_terms': ['2', 'COURSES']},
 {'id': 46,
  'category': 'Examinations',
  'question': 'How is performance in each course evaluated?',
  'expected_terms': ['CONTINUOUS ASSESSMENT', 'END SEMESTER EXAMINATION'],
  'answer_terms': ['CONTINUOUS ASSESSMENT', 'END SEMESTER']},
 {'id': 47,
  'category': 'Examinations',
  'question': 'What is the maximum mark for each course?',
  'expected_terms': ['100 MARKS'],
  'answer_terms': ['100']},
 {'id': 48,
  'category': 'Examinations',
  'question': 'How long is the normal End Semester Examination for theory courses?',
  'expected_terms': ['3 HOURS'],
  'answer_terms': ['3 HOURS']},
 {'id': 49,
  'category': 'Examinations',
  'question': 'When are odd-semester End Semester Examinations normally conducted?',
  'expected_terms': ['NOVEMBER / DECEMBER'],
  'answer_terms': ['NOVEMBER', 'DECEMBER']},
 {'id': 50,
  'category': 'Examinations',
  'question': 'When are even-semester End Semester Examinations normally conducted?',
  'expected_terms': ['APRIL / MAY'],
  'answer_terms': ['APRIL', 'MAY']},
 {'id': 51,
  'category': 'Examinations',
  'question': 'Who appoints internal and external examiners for End Semester Examinations?',
  'expected_terms': ['CONTROLLER OF EXAMINATIONS'],
  'answer_terms': ['CONTROLLER OF EXAMINATIONS']},
 {'id': 52,
  'category': 'Examinations',
  'question': 'What minimum total percentage is required to pass a course?',
  'expected_terms': ['50%'],
  'answer_terms': ['50%']},
 {'id': 53,
  'category': 'Examinations',
  'question': 'What minimum percentage in the End Semester Examination is required for passing a course?',
  'expected_terms': ['45%'],
  'answer_terms': ['45%']},
 {'id': 54,
  'category': 'Examinations',
  'question': 'Where can a student submit an application for revaluation?',
  'expected_terms': ['COE OFFICE', 'CONTROLLER OF EXAMINATIONS'],
  'answer_terms': ['COE']},
 {'id': 55,
  'category': 'Examinations',
  'question': 'How many working days before an End Semester Examination is the normal withdrawal application deadline?',
  'expected_terms': ['TEN WORKING DAYS'],
  'answer_terms': ['TEN', 'WORKING DAYS']},
 {'id': 56,
  'category': 'Fees/administration',
  'question': 'Who heads the Administrative Department of the college?',
  'expected_terms': ['ADMINISTRATIVE OFFICER'],
  'answer_terms': ['ADMINISTRATIVE OFFICER']},
 {'id': 57,
  'category': 'Fees/administration',
  'question': 'Which office handles student service, HR, purchase and dispatch?',
  'expected_terms': ['ADMINISTRATIVE OFFICE'],
  'answer_terms': ['ADMINISTRATIVE OFFICE']},
 {'id': 58,
  'category': 'Fees/administration',
  'question': 'Can students contact the Administrative Office for fee-related queries?',
  'expected_terms': ['FEE RELATED QUERIES'],
  'answer_terms': ['FEE']},
 {'id': 59,
  'category': 'Fees/administration',
  'question': 'Can students contact the Administrative Office for identity cards?',
  'expected_terms': ['IDENTITY CARD'],
  'answer_terms': ['IDENTITY CARD']},
 {'id': 60,
  'category': 'Fees/administration',
  'question': 'Can students contact the Administrative Office for bonafide certificates?',
  'expected_terms': ['BONAFIDE CERTIFICATE'],
  'answer_terms': ['BONAFIDE']},
 {'id': 61,
  'category': 'Fees/administration',
  'question': 'Can students contact the Administrative Office for bus passes?',
  'expected_terms': ['BUS PASS'],
  'answer_terms': ['BUS PASS']},
 {'id': 62,
  'category': 'Fees/administration',
  'question': 'Can students contact the Administrative Office for hall tickets?',
  'expected_terms': ['HALL TICKET'],
  'answer_terms': ['HALL TICKET']},
 {'id': 63,
 'category': 'Fees/administration',
 'question': 'What is the name of the ERP software used for college administration?',
 'expected_terms': ['DIGICAMPUS'],
 'answer_terms': ['DIGICAMPUS']},
 {'id': 64,
  'category': 'Fees/administration',
  'question': 'Which ERP modules include attendance, library, transport, examinations and hostel?',
  'expected_terms': ['ATTENDANCE', 'LIBRARY', 'TRANSPORT', 'EXAMINATIONS', 'HOSTEL'],
  'answer_terms': ['ATTENDANCE', 'LIBRARY', 'TRANSPORT']},
 {'id': 65,
  'category': 'Fees/administration',
  'question': 'Can the REC Unified Portal generate fee certificates?',
  'expected_terms': ['FEE CERTIFICATES'],
  'answer_terms': ['FEE CERTIFICATES']},
 {'id': 66,
  'category': 'Facilities/hostel',
  'question': 'How many boys can the college hostels accommodate?',
  'expected_terms': ['2000 BOYS', '2000'],
  'answer_terms': ['2000']},
 {'id': 67,
  'category': 'Facilities/hostel',
  'question': 'How many girls can the college hostels accommodate?',
  'expected_terms': ['1200 GIRLS', '1200'],
  'answer_terms': ['1200']},
 {'id': 68,
  'category': 'Facilities/hostel',
  'question': 'Are separate hostels available for boys and girls?',
  'expected_terms': ['SEPARATE HOSTELS'],
  'answer_terms': ['SEPARATE', 'HOSTELS']},
 {'id': 69,
  'category': 'Facilities/hostel',
  'question': 'Who is responsible for allocating hostel rooms?',
  'expected_terms': ['WARDEN', 'PRINCIPAL'],
  'answer_terms': ['WARDEN']},
 {'id': 70,
  'category': 'Facilities/hostel',
  'question': 'What type of food is served in the hostel mess?',
  'expected_terms': ['VEGETARIAN FOOD'],
  'answer_terms': ['VEGETARIAN']},
 {'id': 71,
  'category': 'Facilities/hostel',
  'question': 'When are hostel visitors allowed on holidays?',
  'expected_terms': ['11.00 A.M. TO 6.00 P.M.', '11 AM', '6 PM'],
  'answer_terms': ['11', '6']},
 {'id': 72,
  'category': 'Facilities/hostel',
  'question': 'How many books are held by the central library according to the library policy?',
  'expected_terms': ['61,000 BOOKS', '61000'],
  'answer_terms': ['61000', 'BOOKS']},
 {'id': 73,
  'category': 'Facilities/hostel',
  'question': 'Does the central library provide remote access to e-resources?',
  'expected_terms': ['REMOTE ACCESS', 'E-RESOURCES'],
  'answer_terms': ['REMOTE', 'E-RESOURCES']},
 {'id': 74,
  'category': 'Facilities/hostel',
  'question': 'What facility is available in the library for digital resources?',
  'expected_terms': ['DIGITAL LIBRARY'],
  'answer_terms': ['DIGITAL LIBRARY']},
 {'id': 75,
  'category': 'Facilities/hostel',
  'question': 'What sports facilities are maintained by the college sports complex?',
  'expected_terms': ['PLAY FIELDS', 'COURTS', 'TRACKS', 'GYM'],
  'answer_terms': ['PLAY FIELDS', 'COURTS']},
 {'id': 76,
  'category': 'Placements',
  'question': 'Does Rajalakshmi Engineering College have a dedicated Training and Placement Cell?',
  'expected_terms': ['TRAINING AND PLACEMENT CELL'],
  'answer_terms': ['PLACEMENT']},
 {'id': 77,
  'category': 'Placements',
  'question': 'What is the purpose of the Training and Placement Cell?',
  'expected_terms': ['TRAINING', 'EMPLOYMENT'],
  'answer_terms': ['TRAINING']},
 {'id': 78,
  'category': 'Placements',
  'question': 'Can an eligible student apply for a company visiting the campus?',
  'expected_terms': ['ELIGIBLE', 'COMPANY'],
  'answer_terms': ['ELIGIBLE', 'COMPANY']},
 {'id': 79,
  'category': 'Placements',
  'question': 'How many offers were reported for the 2021-2025 placement batch as of 12.09.2025?',
  'expected_terms': ['1941+', '1941'],
  'answer_terms': ['1941']},
 {'id': 80,
  'category': 'Placements',
  'question': 'Name one company listed among the placement recruiters.',
  'expected_terms': ['ACCENTURE', 'AMAZON', 'COGNIZANT', 'BOSCH'],
  'answer_terms': ['ACCENTURE']},
 {'id': 81,
  'category': 'Placements',
  'question': 'What is the salary threshold for a Super Dream offer under the placement policy?',
  'expected_terms': ['6.5 LPA', 'SUPER DREAM'],
  'answer_terms': ['6.5 LPA']},
 {'id': 82,
  'category': 'Placements',
  'question': 'What salary is specified for a Dream offer under the placement policy?',
  'expected_terms': ['4.5 LPA', 'DREAM'],
  'answer_terms': ['4.5 LPA']},
 {'id': 83,
  'category': 'Placements',
  'question': 'What salary range is described as a General offer?',
  'expected_terms': ['LESS THAN 4.5 LPA', 'GENERAL'],
  'answer_terms': ['4.5 LPA']},
 {'id': 84,
  'category': 'Placements',
  'question': 'Can a student with a General offer apply for another Dream, Super Dream or Core offer?',
  'expected_terms': ['GENERAL OFFER', 'DREAM', 'SUPER DREAM', 'CORE'],
  'answer_terms': ['GENERAL', 'DREAM']},
 {'id': 85,
  'category': 'Placements',
  'question': 'What training does the placement centre provide to improve employability?',
  'expected_terms': ['TRAINING', 'VALUE ADDED PROGRAMS'],
  'answer_terms': ['TRAINING']},
 {'id': 86,
  'category': 'Campus/events',
  'question': 'Where does Rajalakshmi Engineering College publish its latest events and updates?',
  'expected_terms': ['LATEST EVENTS', 'UPDATES'],
  'answer_terms': ['EVENTS']},
 {'id': 87,
  'category': 'Campus/events',
  'question': 'Which student-life section covers athletics?',
  'expected_terms': ['ATHLETICS'],
  'answer_terms': ['ATHLETICS']},
 {'id': 88,
  'category': 'Campus/events',
  'question': 'Which student-life section covers health care?',
  'expected_terms': ['HEALTH CARE'],
  'answer_terms': ['HEALTH CARE']},
 {'id': 89,
  'category': 'Campus/events',
  'question': 'Which student-life section covers technical societies?',
  'expected_terms': ['TECHNICAL SOCIETIES'],
  'answer_terms': ['TECHNICAL SOCIETIES']},
 {'id': 90,
  'category': 'Campus/events',
  'question': 'Which student-life section covers dining facilities?',
  'expected_terms': ['DINING'],
  'answer_terms': ['DINING']},
 {'id': 91,
  'category': 'General institutional information',
  'question': 'In which year was Rajalakshmi Engineering College established?',
  'expected_terms': ['1997'],
  'answer_terms': ['1997']},
 {'id': 92,
  'category': 'General institutional information',
  'question': 'Is Rajalakshmi Engineering College an autonomous institution affiliated to Anna University?',
  'expected_terms': ['AUTONOMOUS', 'ANNA UNIVERSITY'],
  'answer_terms': ['AUTONOMOUS', 'ANNA UNIVERSITY']},
 {'id': 93,
  'category': 'General institutional information',
  'question': 'Where is the college campus located?',
  'expected_terms': ['THANDALAM', 'CHENNAI', '602105'],
  'answer_terms': ['THANDALAM', 'CHENNAI']},
 {'id': 94,
  'category': 'General institutional information',
  'question': 'How many undergraduate programmes does the college currently list on its admissions information?',
  'expected_terms': ['19 UNDERGRADUATE', '19'],
  'answer_terms': ['19']},
 {'id': 95,
  'category': 'General institutional information',
  'question': 'What accreditation grade is reported for Rajalakshmi Engineering College by NAAC?',
  'expected_terms': ['A++'],
  'answer_terms': ['A++']},
 {'id': 96,
  'category': 'Unsupported/out-of-scope',
  'question': 'What is the weather today in Chennai?',
  'expected_terms': [],
  'answer_terms': [],
  'negative': True},
 {'id': 97,
  'category': 'Unsupported/out-of-scope',
  'question': 'What is the current salary of a CSE faculty member at the college?',
  'expected_terms': [],
  'answer_terms': [],
  'negative': True},
 {'id': 98,
  'category': 'Unsupported/out-of-scope',
  'question': 'What is my personal attendance percentage for this semester?',
  'expected_terms': [],
  'answer_terms': [],
  'negative': True},
 {'id': 99,
  'category': 'Unsupported/out-of-scope',
  'question': 'What is my current outstanding fee amount?',
  'expected_terms': [],
  'answer_terms': [],
  'negative': True},
 {'id': 100,
  'category': 'Unsupported/out-of-scope',
  'question': 'What are my individual semester examination results?',
  'expected_terms': [],
  'answer_terms': [],
  'negative': True}]


# ============================================================
# TEXT NORMALIZATION
# ============================================================


def normalize(text: str) -> str:
    """
    Normalize text for answer and retrieval evaluation.
    Treat '&' and 'AND' as equivalent.
    """

    text = str(text or "").upper()

    # Normalize equivalent conjunctions before punctuation handling.
    text = re.sub(r"\s*&\s*", " AND ", text)

    # Normalize dash variants.
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# GET RETRIEVED TEXT
# ============================================================

def get_retrieved_text(results) -> str:

    pieces = []

    for item in results:

        if isinstance(item, dict):

            text = item.get(
                "text",
                "",
            )

            if text:

                pieces.append(
                    str(text)
                )

        else:

            pieces.append(
                str(item)
            )

    return "\n".join(
        pieces
    )


# ============================================================
# RETRIEVAL CHECK
# ============================================================

def retrieval_hit(
    retrieved_text: str,
    expected_terms: list[str],
) -> bool:

    if not expected_terms:

        return False

    normalized_text = normalize(
        retrieved_text
    )

    for term in expected_terms:

        if normalize(term) in normalized_text:

            return True

    return False


# ============================================================
# ANSWER CHECK
# ============================================================

def answer_correct(
    answer: str,
    expected_terms: list[str],
) -> bool:

    if not expected_terms:

        return False

    normalized_answer = normalize(
        answer
    )

    # --------------------------------------------------------
    # For short factual answers, at least one expected
    # answer term is enough.
    #
    # For multi-term answers, we require at least half.
    # --------------------------------------------------------

    matched = 0

    for term in expected_terms:

        if normalize(term) in normalized_answer:

            matched += 1

    required = max(
        1,
        (len(expected_terms) + 1) // 2,
    )

    return matched >= required


# ============================================================
# NEGATIVE ANSWER CHECK
# ============================================================

def refusal_detected(
    answer: str,
) -> bool:

    text = normalize(
        answer
    )

    refusal_patterns = [
        "I DON'T HAVE ENOUGH INFORMATION",
        "I DO NOT HAVE ENOUGH INFORMATION",
        "NOT IN OUR",
        "NOT AVAILABLE",
        "I'M SORRY",
        "I AM SORRY",
        "PLEASE CONTACT RECEPTION",
        "NO INFORMATION",
        "CANNOT ANSWER",
        "CAN'T ANSWER",
    ]

    for pattern in refusal_patterns:

        if pattern in text:

            return True

    return False


# ============================================================
# MAIN EVALUATION
# ============================================================

def main() -> None:

    print()
    print("=" * 75)
    print("REC AI RECEPTIONIST - RAG ACCURACY EVALUATION")
    print("=" * 75)

    print()
    print("[INFO] Project root:")
    print(PROJECT_ROOT)

    print()
    print(
        f"[INFO] Test questions: "
        f"{len(TEST_CASES)}"
    )

    print()
    print("[STARTUP] Loading Retriever...")

    retriever = Retriever()

    print(
        "[STARTUP] Retriever ready."
    )

    print()
    print("[STARTUP] Loading RAG service...")

    rag = RAGService()

    print(
        "[STARTUP] RAG service ready."
    )

    print()
    print("=" * 75)
    print("RUNNING EVALUATION")
    print("=" * 75)

    results = []

    retrieval_correct = 0
    answer_correct_count = 0

    normal_questions = 0

    negative_total = 0
    negative_correct = 0

    total_time = 0.0

    # ========================================================
    # RUN EACH TEST
    # ========================================================

    for index, test in enumerate(
        TEST_CASES,
        start=1,
    ):

        question_id = test["id"]
        category = test["category"]
        question = test["question"]

        is_negative = test.get(
            "negative",
            False,
        )

        print()
        print("-" * 75)
        print(
            f"[TEST {index}/{len(TEST_CASES)}] "
            f"{category}"
        )
        print(
            f"Question: {question}"
        )

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        start_time = time.perf_counter()

        try:

            retrieved = retriever.search(
                question,
                top_k=5,
            )

            retrieved_text = get_retrieved_text(
                retrieved
            )

            retrieval_ok = retrieval_hit(
                retrieved_text,
                test.get(
                    "expected_terms",
                    [],
                ),
            )

        except Exception as exc:

            print(
                f"[RETRIEVAL ERROR] {exc}"
            )

            retrieved = []
            retrieved_text = ""
            retrieval_ok = False

        # ----------------------------------------------------
        # RAG ANSWER
        # ----------------------------------------------------

        answer = ""

        try:

            rag_result = rag.ask(
                question
            )

            if isinstance(
                rag_result,
                dict,
            ):

                answer = str(
                    rag_result.get(
                        "answer",
                        "",
                    )
                ).strip()

            else:

                answer = str(
                    rag_result
                ).strip()

        except Exception as exc:

            print(
                f"[RAG ERROR] {exc}"
            )

            answer = ""

        elapsed = (
            time.perf_counter()
            - start_time
        )

        total_time += elapsed

        # ----------------------------------------------------
        # NORMAL QUESTION
        # ----------------------------------------------------

        if not is_negative:

            normal_questions += 1

            if retrieval_ok:

                retrieval_correct += 1

            answer_ok = answer_correct(
                answer,
                test.get(
                    "answer_terms",
                    [],
                ),
            )

            if answer_ok:

                answer_correct_count += 1

        # ----------------------------------------------------
        # NEGATIVE QUESTION
        # ----------------------------------------------------

        else:

            negative_total += 1

            answer_ok = refusal_detected(
                answer
            )

            if answer_ok:

                negative_correct += 1

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print()

        print(
            "[RETRIEVAL] "
            + (
                "PASS"
                if retrieval_ok
                else "FAIL"
            )
        )

        print(
            "[ANSWER]"
        )

        print(
            answer
            if answer
            else "[NO ANSWER]"
        )

        if is_negative:

            print()

            print(
                "[REFUSAL TEST] "
                + (
                    "PASS"
                    if answer_ok
                    else "FAIL"
                )
            )

        else:

            print()

            print(
                "[ANSWER CORRECTNESS] "
                + (
                    "PASS"
                    if answer_ok
                    else "FAIL"
                )
            )

        print(
            f"[TIME] {elapsed:.2f}s"
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append(
            {
                "id": question_id,
                "category": category,
                "question": question,
                "retrieval_correct": (
                    "YES"
                    if retrieval_ok
                    else "NO"
                ),
                "answer": answer,
                "answer_correct": (
                    "YES"
                    if answer_ok
                    else "NO"
                ),
                "time_seconds": f"{elapsed:.2f}",
            }
        )

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    retrieval_accuracy = (
        retrieval_correct
        / normal_questions
        * 100
        if normal_questions
        else 0
    )

    answer_accuracy = (
        answer_correct_count
        / normal_questions
        * 100
        if normal_questions
        else 0
    )

    refusal_accuracy = (
        negative_correct
        / negative_total
        * 100
        if negative_total
        else 0
    )

    average_time = (
        total_time
        / len(TEST_CASES)
        if TEST_CASES
        else 0
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        CSV_REPORT,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        fieldnames = [
            "id",
            "category",
            "question",
            "retrieval_correct",
            "answer",
            "answer_correct",
            "time_seconds",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ========================================================
    # SAVE AGGREGATED METRICS
    # ========================================================
    # These values are calculated from this evaluation run.
    # visualize_results.py reads this file; it does not
    # recalculate the Direct RAG metrics.

    metric_rows = [
        {
            "metric": "Retrieval Hit@5",
            "accuracy_percent": round(retrieval_accuracy, 2),
            "sample_count": normal_questions,
        },
        {
            "metric": "Answer Accuracy",
            "accuracy_percent": round(answer_accuracy, 2),
            "sample_count": normal_questions,
        },
        {
            "metric": "Refusal Accuracy",
            "accuracy_percent": round(refusal_accuracy, 2),
            "sample_count": negative_total,
        },
        {
            "metric": "Overall Test Questions",
            "accuracy_percent": "",
            "sample_count": len(TEST_CASES),
        },
    ]

    with open(
        METRICS_REPORT,
        "w",
        newline="",
        encoding="utf-8",
    ) as metrics_file:
        fieldnames = [
            "metric",
            "accuracy_percent",
            "sample_count",
        ]
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"[METRICS REPORT] {METRICS_REPORT}")

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print()
    print("=" * 75)
    print("FINAL EVALUATION RESULT")
    print("=" * 75)

    print()

    print(
        f"Normal test questions : "
        f"{normal_questions}"
    )

    print(
        f"Retrieval correct     : "
        f"{retrieval_correct}"
    )

    print(
        f"Answer correct        : "
        f"{answer_correct_count}"
    )

    print()

    print(
        f"Retrieval Hit@5       : "
        f"{retrieval_accuracy:.2f}%"
    )

    print(
        f"Answer Accuracy       : "
        f"{answer_accuracy:.2f}%"
    )

    print()

    print(
        f"Negative tests        : "
        f"{negative_total}"
    )

    print(
        f"Correct refusals      : "
        f"{negative_correct}"
    )

    print(
        f"Refusal Accuracy      : "
        f"{refusal_accuracy:.2f}%"
    )

    print()

    print(
        f"Average processing    : "
        f"{average_time:.2f} seconds"
    )

    print()

    print(
        f"[QUERY REPORT] "
        f"{CSV_REPORT}"
    )

    print(
        f"[METRICS REPORT] "
        f"{METRICS_REPORT}"
    )

    print()

    print("=" * 75)
    print("EVALUATION COMPLETE")
    print("=" * 75)

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()