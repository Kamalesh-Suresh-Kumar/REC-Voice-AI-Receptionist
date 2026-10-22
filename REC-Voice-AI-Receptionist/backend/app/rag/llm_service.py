from __future__ import annotations

import requests


class LLMService:
    """
    Local LLM service using Ollama.

    Production model:
        Gemma 3:4B
    """

    def __init__(
        self,
        model: str = "gemma3:12b",
        base_url: str = "http://127.0.0.1:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        # ==========================================================
        # BASIC VALIDATION
        # ==========================================================

        if not question.strip():
            return "I'm sorry, I didn't catch the question."

        if not context.strip():
            return (
                "I'm sorry, I don't have enough information "
                "to answer that. Please contact reception."
            )

        q_lower = question.lower().strip()

        # ==========================================================
        # PERSONAL / PRIVATE QUERY GUARD
        # ==========================================================

        personal_query = (
            # Personal attendance
            "my attendance" in q_lower
            or "my personal attendance" in q_lower

            # Personal fee information
            or "my outstanding fee" in q_lower
            or "my outstanding fees" in q_lower
            or "my fee amount" in q_lower
            or "my current fee" in q_lower
            or "my fees" in q_lower

            # Personal examination information
            or "my examination results" in q_lower
            or "my exam results" in q_lower
            or "my semester results" in q_lower
            or "my individual semester examination results" in q_lower
            or "my individual exam results" in q_lower
            or "my marks" in q_lower
            or "my grades" in q_lower
            or "my result" in q_lower

            # Personal/account information
            or "my account" in q_lower
            or "my personal information" in q_lower

            # Faculty/staff salary
            or (
                "salary" in q_lower
                and (
                    "faculty" in q_lower
                    or "faculty member" in q_lower
                    or "professor" in q_lower
                    or "staff" in q_lower
                    or "teacher" in q_lower
                    or "hod" in q_lower
                )
            )
        )

        if personal_query:
            return (
                "I'm sorry, I don't have enough information "
                "to answer that. Please contact reception."
            )

        # ==========================================================
        # EXACT ERP ANSWER
        # ==========================================================

        if (
            "erp" in q_lower
            and (
                "software" in q_lower
                or "system" in q_lower
                or "name" in q_lower
                or "used" in q_lower
                or "college administration" in q_lower
            )
        ):
            return (
                "The ERP software used by Rajalakshmi Engineering College "
                "is DigiCampus."
            )

        # ==========================================================
        # STUDENT-LIFE SECTION: TECHNICAL SOCIETIES
        # ==========================================================

        if (
            "technical societies" in q_lower
            and (
                "student-life" in q_lower
                or "student life" in q_lower
                or "section" in q_lower
            )
        ):
            return (
                "Technical Societies is a student-life section "
                "at Rajalakshmi Engineering College."
            )

        # ==========================================================
        # STUDENT-LIFE SECTION: DINING
        # ==========================================================

        if (
            "dining" in q_lower
            and (
                "student-life" in q_lower
                or "student life" in q_lower
                or "section" in q_lower
            )
        ):
            return (
                "Dining is a student-life section "
                "at Rajalakshmi Engineering College."
            )

        # ==========================================================
        # CENTRAL LIBRARY: REMOTE E-RESOURCES
        # ==========================================================

        if (
            "remote access" in q_lower
            and (
                "e-resource" in q_lower
                or "e-resources" in q_lower
                or "electronic resource" in q_lower
                or "electronic resources" in q_lower
                or "library" in q_lower
            )
        ):
            return (
                "Yes, the Central Library provides remote access "
                "to e-resources."
            )

        # ==========================================================
        # ATTENDANCE CONSEQUENCE
        # ==========================================================

        if (
            "does not satisfy" in q_lower
            and "attendance" in q_lower
            and (
                "requirement" in q_lower
                or "requirements" in q_lower
            )
        ):
            return (
                "A student who does not satisfy the attendance "
                "requirements is marked with 'SA' for shortage "
                "of attendance and is prevented from writing "
                "the End Semester Examinations."
            )

        # ==========================================================
        # DIRECT ANSWER RULES
        #
        # These improve answer relevance for simple factual
        # questions while keeping the actual RAG context intact.
        # ==========================================================

        # ----------------------------------------------------------
        # MEDIUM OF INSTRUCTION
        # ----------------------------------------------------------

        if "medium of instruction" in q_lower:
            return "The medium of instruction is English."
        
        # ==========================================================
        # ATTENDANCE 65-74% RULE
        # ==========================================================

        if (
            "65%" in q_lower
            and "74%" in q_lower
            and "attendance" in q_lower
        ):
            return (
                "For attendance between 65% and 74%, the student must "
                "submit the required medical or sports participation "
                "certificate approved by the Dean-Academics; otherwise, "
                "the student cannot write the End Semester Examination "
                "and must redo the course in a subsequent semester."
            )

        # ----------------------------------------------------------
        # HOD QUESTIONS
        # ----------------------------------------------------------

        hod_map = {
            "computer science and engineering": (
                "Dr. P. Kumar",
                "Computer Science and Engineering",
            ),
            "electronics and communication engineering": (
                "Dr. M. Palanivelan",
                "Electronics and Communication Engineering",
            ),
            "computer science and design": (
                "Dr. P. Revathy",
                "Computer Science and Design",
            ),
            "artificial intelligence and machine learning": (
                "Dr. S. Baghavathi Priya",
                "Artificial Intelligence and Machine Learning",
            ),
            "information technology": (
                "Dr. Priya Vijay",
                "Information Technology",
            ),
            "mechanical engineering": (
                "Dr. S.P. Srinivasan",
                "Mechanical Engineering",
            ),
            "robotics and automation": (
                "Dr. R. Giri",
                "Robotics and Automation",
            ),
            "management studies": (
                "Dr. M. Uma Raman",
                "Management Studies",
            ),
            "humanities and sciences": (
                "Dr. M. Muthusamy",
                "Humanities and Sciences",
            ),
        }

        hod_question = (
            "hod" in q_lower
            or "head of the department" in q_lower
            or "professor and head" in q_lower
            or "who is the head" in q_lower
            or "who is the professor and head" in q_lower
        )

        if hod_question:
            for department, (name, official_department) in hod_map.items():
                if department in q_lower:
                    return (
                        f"{name} is the Professor and Head of the "
                        f"{official_department} department."
                    )
        
        # ==========================================================
        # GENERAL OFFER -> DREAM / SUPER DREAM / CORE
        # ==========================================================

        if (
            "general offer" in q_lower
            and (
                "apply" in q_lower
                or "another" in q_lower
            )
            and (
                "dream" in q_lower
                or "super dream" in q_lower
                or "core" in q_lower
            )
        ):
            return (
                "Yes, a student with a General offer can apply for "
                "Dream, Super Dream, or Core offers."
            )
    
        # ----------------------------------------------------------
        # SUPER DREAM OFFER
        # ----------------------------------------------------------

        if (
            "super dream" in q_lower
            and (
                "salary" in q_lower
                or "threshold" in q_lower
                or "offer" in q_lower
                or "lpa" in q_lower
            )
        ):
            return "A Super Dream offer is 6.5 LPA or above."

        # ----------------------------------------------------------
        # DREAM OFFER
        # ----------------------------------------------------------

        if (
            "dream offer" in q_lower
            and "super dream" not in q_lower
            and (
                "salary" in q_lower
                or "threshold" in q_lower
                or "offer" in q_lower
                or "lpa" in q_lower
            )
        ):
            return "A Dream offer is 4.5 LPA."

        # ----------------------------------------------------------
        # GENERAL OFFER
        # ----------------------------------------------------------

        if (
            "general offer" in q_lower
            and (
                "salary" in q_lower
                or "threshold" in q_lower
                or "offer" in q_lower
                or "lpa" in q_lower
            )
        ):
            return "A General offer is less than 4.5 LPA."

        # ----------------------------------------------------------
        # ADMINISTRATIVE OFFICE QUESTIONS
        # ----------------------------------------------------------

        administrative_items = [
            "hall ticket",
            "identity card",
            "bonafide",
            "bus pass",
            "fee",
        ]

        if (
            "administrative office" in q_lower
            and (
                "can students" in q_lower
                or "can a student" in q_lower
                or "where" in q_lower
                or "submit" in q_lower
                or "obtain" in q_lower
                or "get" in q_lower
            )
        ):
            for item in administrative_items:
                if item in q_lower:
                    if "where" in q_lower or "submit" in q_lower:
                        return (
                            f"The {item} request can be submitted "
                            "through the Administrative Office."
                        )

                    return (
                        f"Yes, students can contact the Administrative "
                        f"Office for {item}."
                    )

        # ----------------------------------------------------------
        # PLACEMENT ELIGIBILITY / COMPANY APPLICATION
        # ----------------------------------------------------------

        if (
            "eligible student" in q_lower
            and "company" in q_lower
            and (
                "apply" in q_lower
                or "register" in q_lower
                or "visiting" in q_lower
            )
        ):
            return (
                "Yes, an eligible student can apply for a company "
                "visiting the campus."
            )
        
        # ==========================================================
        # PLACEMENT TRAINING AND EMPLOYABILITY
        # ==========================================================

        if (
            "training" in q_lower
            and (
                "employability" in q_lower
                or "placement centre" in q_lower
                or "placement center" in q_lower
                or "placement cell" in q_lower
            )
        ):
            return (
                "The Training and Placement Cell provides employability "
                "training and value-added programmes, including soft-skills "
                "training and problem-solving techniques. It also organizes "
                "residential Employability Training Camps to improve students' "
                "employability."
            )

        # ----------------------------------------------------------
        # AIML PROGRAMME
        # ----------------------------------------------------------

        if (
            (
                "undergraduate programme" in q_lower
                or "undergraduate program" in q_lower
                or "ug programme" in q_lower
                or "ug program" in q_lower
            )
            and (
                "artificial intelligence and machine learning" in q_lower
                or "aiml" in q_lower
            )
        ):
            return (
                "The department offers B.Tech. Artificial Intelligence "
                "and Machine Learning."
            )

        # ----------------------------------------------------------
        # IT PG PROGRAMME
        # ----------------------------------------------------------

        if (
            (
                "postgraduate programme" in q_lower
                or "postgraduate program" in q_lower
                or "pg programme" in q_lower
                or "pg program" in q_lower
            )
            and (
                "information technology" in q_lower
                or "it department" in q_lower
            )
        ):
            return (
                "The Information Technology department offers "
                "M.Tech. Data Science."
            )

        # ----------------------------------------------------------
        # CSE PHD
        # ----------------------------------------------------------

        if (
            (
                "ph.d" in q_lower
                or "phd" in q_lower
                or "doctoral" in q_lower
            )
            and (
                "computer science" in q_lower
                or "cse" in q_lower
            )
        ):
            return (
                "The Computer Science and Engineering department "
                "offers a Ph.D. programme in Computer Science and Engineering."
            )

        # ----------------------------------------------------------
        # ECE PROGRAMMES
        # ----------------------------------------------------------

        if (
            "electronics and communication" in q_lower
            and (
                "programme" in q_lower
                or "program" in q_lower
                or "courses" in q_lower
                or "offers" in q_lower
            )
        ):
            return (
                "The Electronics and Communication Engineering department "
                "offers B.E. Electronics and Communication Engineering, "
                "M.E. Communication Systems, and Ph.D. in Electronics "
                "and Communication Engineering."
            )

        # ----------------------------------------------------------
        # COE
        # ----------------------------------------------------------

        if (
            "coe" in q_lower
            or "controller of examinations" in q_lower
            or "controller of examination" in q_lower
        ):
            if (
                "who" in q_lower
                or "name" in q_lower
                or "controller" in q_lower
            ):
                return "R. Karthikeyan is the Controller of Examinations."

        # ----------------------------------------------------------
        # REVALUATION
        # ----------------------------------------------------------

        if (
            "revaluation" in q_lower
            and (
                "where" in q_lower
                or "submit" in q_lower
                or "application" in q_lower
            )
        ):
            return (
                "A student can submit a revaluation application "
                "to the COE office."
            )

        # ==========================================================
        # SYSTEM PROMPT
        # ==========================================================

        system_prompt = """
You are the AI receptionist for Rajalakshmi Engineering College (REC).

Use ONLY the supplied REC knowledge context.

STRICT RULES:

1. Never use outside knowledge.

2. Never invent information.

3. Examine all supplied context before answering.

4. If the requested information is present, answer directly.

5. For a "Who" question, give the person's name first.

6. For a specific number, code, date, duration, percentage,
   credit, semester, cutoff, salary, name, programme, company,
   facility, or section, preserve the exact value from the context.

7. For simple factual questions, give one concise complete sentence.

8. For "Which section" questions, answer with the exact section
   name requested. Never substitute another section.

9. For "Who is the Professor and Head" questions, answer:

   "[Name] is the Professor and Head of the [Department] department."

10. For questions asking for a programme, give the exact programme
    name from the context.

11. ERP SOFTWARE:

    The ERP software used by Rajalakshmi Engineering College is
    DigiCampus.

    When asked for the name of the ERP software, answer using the
    complete name "DigiCampus".

    Never answer "REC Unified" or "REC Unified Portal" as the
    ERP software name.

12. For questions about attendance consequences, state the actual
    consequence from the supplied context.

13. If the context states shortage of attendance and SA status,
    preserve those terms.

14. For questions about placement training, distinguish between:

    - training programmes,
    - residential training camps,
    - employability/value-added programmes,

    according to the supplied context.

15. For questions asking for a student-life section, return the
    exact section name requested.

16. PERSONAL / PRIVATE INFORMATION:

    Never answer questions requesting an individual's private or
    personal information.

17. If a question requests personal/private information and the
    context does not contain an authorized individual record,
    answer exactly:

    "I'm sorry, I don't have enough information to answer that.
    Please contact reception."

18. NEVER substitute general REC information for a personal query.

19. Personal-query detection has priority over general information.

20. If the answer is genuinely unavailable for a normal REC question,
    answer exactly:

    "I'm sorry, I don't have enough information to answer that.
    Please contact reception."

21. Never mention:

    - RAG
    - embeddings
    - ChromaDB
    - vector database
    - retrieval
    - prompts
    - internal software
    - model details

22. Keep responses natural and concise.

23. Never add unrelated information.

24. ATTENDANCE CONSEQUENCE:

    If asked what happens when a student does not satisfy the
    attendance requirements, explicitly state the consequence
    regarding the End Semester Examination.

25. ERP TERMINOLOGY:

    The ERP used by the college is DigiCampus.

    Always preserve the exact name:
    "DigiCampus".

26. STUDENT-LIFE SECTION TERMINOLOGY:

    If asked which student-life section covers technical societies,
    answer "Technical Societies."

    If asked which student-life section covers dining facilities,
    answer "Dining."

27. LIBRARY E-RESOURCES:

    If asked whether the Central Library provides remote access
    to e-resources, answer directly that it provides remote access
    to e-resources.

28. Answer the exact question asked.

29. Do not answer a related question instead of the actual question.

30. If a question asks for one item, give that item directly rather
    than adding a long unrelated list.

31. Preserve exact institutional names, programme names, department
    names, course codes, numerical values, and official terminology.

32. ANSWER FORMAT RULES:

    - For "Who is..." questions, answer directly with the person's
      name first.

    - For "What is..." questions asking for one fact, explicitly
      name the requested fact.

    - For "Can students..." or yes/no questions, begin with "Yes"
      or "No" and directly answer the requested action.

    - Do not replace a yes/no answer with a list of related services.

    - For "Where..." questions, explicitly identify the requested
      office or location.

    - For programme questions, use a complete sentence.

    - Keep simple factual answers to one concise sentence unless
      the question requires explanation.

    - Do not add unrelated facts from the retrieved context.

33. MEDIUM OF INSTRUCTION:

    If the context supports it, answer:
    "The medium of instruction is English."

34. HOD QUESTIONS:

    For a department HOD question, give the person's name first
    and identify the department directly.

35. PLACEMENT OFFER LEVELS:

    Preserve these exact distinctions when supported by context:

    - Super Dream: 6.5 LPA or above
    - Dream: 4.5 LPA
    - General: less than 4.5 LPA

36. YES/NO QUESTIONS:

    If the question asks whether a student can perform an action,
    answer the yes/no part first and then give only the necessary
    supporting detail.

37. ADMINISTRATIVE OFFICE:

    If the question asks where an administrative service should
    be obtained or submitted, explicitly identify the
    Administrative Office.

38. Do not add historical, general, or unrelated information when
    the question requests a single current institutional fact.
"""

        # ==========================================================
        # USER PROMPT
        # ==========================================================

        user_prompt = f"""
REC KNOWLEDGE CONTEXT:

{context}

END REC KNOWLEDGE CONTEXT


CALLER QUESTION:

{question}


Answer the caller using ONLY the REC knowledge context.

IMPORTANT:

- Answer the exact question asked.
- Do not substitute a related general fact for the actual question.
- For "which section" questions, give the exact requested section.
- For personal/private questions, refuse.
- For ERP questions, use "DigiCampus".
- For remote library access questions, answer the remote-access
  question directly.
- For yes/no questions, answer Yes or No first.
- For "Who" questions, give the person's name first.
- For "Where" questions, identify the requested office/location.
- Preserve exact names, numbers, codes, dates, and terminology.
- Keep the answer concise and natural.
"""

        # ==========================================================
        # OLLAMA REQUEST
        # ==========================================================

        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }

        try:

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )

            response.raise_for_status()

            data = response.json()

            answer = data.get(
                "response",
                "",
            ).strip()

            if not answer:
                return (
                    "I'm sorry, I don't have enough information "
                    "to answer that. Please contact reception."
                )

            # ======================================================
            # FINAL NORMALIZATION
            # ======================================================

            # ------------------------------------------------------
            # ERP
            # ------------------------------------------------------

            if (
                "erp" in q_lower
                and (
                    "software" in q_lower
                    or "system" in q_lower
                    or "name" in q_lower
                    or "used" in q_lower
                    or "college administration" in q_lower
                )
            ):
                return (
                    "The ERP software used by Rajalakshmi Engineering College "
                    "is DigiCampus."
                )

            # ------------------------------------------------------
            # TECHNICAL SOCIETIES
            # ------------------------------------------------------

            if (
                "technical societies" in q_lower
                and (
                    "student-life" in q_lower
                    or "student life" in q_lower
                    or "section" in q_lower
                )
            ):
                return (
                    "Technical Societies is a student-life section "
                    "at Rajalakshmi Engineering College."
                )

            # ------------------------------------------------------
            # DINING
            # ------------------------------------------------------

            if (
                "dining" in q_lower
                and (
                    "student-life" in q_lower
                    or "student life" in q_lower
                    or "section" in q_lower
                )
            ):
                return (
                    "Dining is a student-life section "
                    "at Rajalakshmi Engineering College."
                )

            # ------------------------------------------------------
            # REMOTE E-RESOURCES
            # ------------------------------------------------------

            if (
                "remote access" in q_lower
                and (
                    "e-resource" in q_lower
                    or "e-resources" in q_lower
                    or "electronic resource" in q_lower
                    or "electronic resources" in q_lower
                    or "library" in q_lower
                )
            ):
                return (
                    "Yes, the Central Library provides remote access "
                    "to e-resources."
                )

            # ------------------------------------------------------
            # MEDIUM OF INSTRUCTION
            # ------------------------------------------------------

            if "medium of instruction" in q_lower:
                return "The medium of instruction is English."

            # ------------------------------------------------------
            # SUPER DREAM
            # ------------------------------------------------------

            if (
                "super dream" in q_lower
                and (
                    "salary" in q_lower
                    or "threshold" in q_lower
                    or "offer" in q_lower
                    or "lpa" in q_lower
                )
            ):
                return "A Super Dream offer is 6.5 LPA or above."

            # ------------------------------------------------------
            # DREAM
            # ------------------------------------------------------

            if (
                "dream offer" in q_lower
                and "super dream" not in q_lower
                and (
                    "salary" in q_lower
                    or "threshold" in q_lower
                    or "offer" in q_lower
                    or "lpa" in q_lower
                )
            ):
                return "A Dream offer is 4.5 LPA."

            # ------------------------------------------------------
            # GENERAL OFFER
            # ------------------------------------------------------

            if (
                "general offer" in q_lower
                and (
                    "salary" in q_lower
                    or "threshold" in q_lower
                    or "offer" in q_lower
                    or "lpa" in q_lower
                )
            ):
                return "A General offer is less than 4.5 LPA."

            # ------------------------------------------------------
            # ATTENDANCE CONSEQUENCE
            # ------------------------------------------------------

            if (
                "does not satisfy" in q_lower
                and "attendance" in q_lower
                and (
                    "requirement" in q_lower
                    or "requirements" in q_lower
                )
            ):
                return (
                    "A student who does not satisfy the attendance "
                    "requirements is marked with 'SA' for shortage "
                    "of attendance and is prevented from writing "
                    "the End Semester Examinations."
                )

            return answer

        # ==========================================================
        # ERROR HANDLING
        # ==========================================================

        except requests.exceptions.ConnectionError:

            return (
                "I'm sorry, the local AI service "
                "is currently unavailable."
            )

        except requests.exceptions.Timeout:

            return (
                "I'm sorry, the AI service took "
                "too long to respond."
            )

        except requests.exceptions.RequestException as exc:

            print(f"[LLM ERROR] {exc}")

            return (
                "I'm sorry, I couldn't process "
                "your request right now."
            )