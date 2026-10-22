from __future__ import annotations

import re
from typing import Any

from .embeddings import EmbeddingModel
from .vector_store import VectorStore


class Retriever:
    """
    Retrieves the most relevant REC knowledge-base
    documents for a user query.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        top_k: int = 5,
    ):
        self.embedding_model = (
            embedding_model or EmbeddingModel()
        )

        self.vector_store = (
            vector_store or VectorStore()
        )

        self.top_k = top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:

        if not query.strip():
            return []

        k = top_k or self.top_k

        # ==================================================
        # 1. SEMANTIC VECTOR SEARCH
        # ==================================================

        query_embedding = self.embedding_model.encode(
            [query]
        )[0]

        candidate_k = max(k * 15, 75)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
        )

        documents_raw = results.get("documents")
        metadatas_raw = results.get("metadatas")
        distances_raw = results.get("distances")

        documents = (
            documents_raw[0]
            if documents_raw
            else []
        )

        metadatas = (
            metadatas_raw[0]
            if metadatas_raw
            else []
        )

        distances = (
            distances_raw[0]
            if distances_raw
            else []
        )

        retrieved: list[dict[str, Any]] = []

        # ==================================================
        # 2. QUERY NORMALIZATION
        # ==================================================

        query_normalized = " ".join(
            query.lower().split()
        )

        query_terms = [
            term.strip(".,?!:;()[]{}\"'")
            for term in query_normalized.split()
            if len(
                term.strip(".,?!:;()[]{}\"'")
            ) >= 3
        ]

        def contains_any(
            text: str,
            terms: list[str],
        ) -> bool:
            return any(
                term in text
                for term in terms
            )

        # ==================================================
        # 3. DEPARTMENT ALIASES
        # ==================================================

        department_aliases = {
            "computer science and engineering": [
                "computer science and engineering",
                "computer science & engineering",
                "cse",
            ],

            "electronics and communication engineering": [
                "electronics and communication engineering",
                "electronics & communication engineering",
                "ece",
            ],

            "information technology": [
                "information technology",
                "information technology department",
                "it department",
            ],

            "artificial intelligence and machine learning": [
                "artificial intelligence and machine learning",
                "artificial intelligence & machine learning",
                "aiml",
            ],

            "computer science and design": [
                "computer science and design",
                "computer science & design",
                "csd",
            ],

            "mechanical engineering": [
                "mechanical engineering",
                "mechanical",
            ],

            "robotics and automation": [
                "robotics and automation",
                "robotics & automation",
                "robotics",
            ],

            "management studies": [
                "management studies",
                "management",
                "mba",
            ],

            "humanities and sciences": [
                "humanities and sciences",
                "humanities & sciences",
                "humanities and science",
                "h&s",
            ],

            "aeronautical engineering": [
                "aeronautical engineering",
                "aeronautical",
            ],

            "automobile engineering": [
                "automobile engineering",
                "automobile",
            ],

            "biomedical engineering": [
                "biomedical engineering",
                "biomedical",
            ],

            "biotechnology": [
                "biotechnology",
            ],

            "chemical engineering": [
                "chemical engineering",
                "chemical",
            ],

            "civil engineering": [
                "civil engineering",
                "civil",
            ],

            "food technology": [
                "food technology",
                "food tech",
            ],

            "artificial intelligence and data science": [
                "artificial intelligence and data science",
                "artificial intelligence & data science",
                "aids",
            ],

            "electrical and electronics engineering": [
                "electrical and electronics engineering",
                "electrical & electronics engineering",
                "eee",
            ],

            "mechatronics engineering": [
                "mechatronics engineering",
                "mechatronics",
            ],
        }

        matched_department = None

        for department, aliases in department_aliases.items():

            if any(
                alias in query_normalized
                for alias in aliases
            ):
                matched_department = department
                break

        # ==================================================
        # 4. INTENTS
        # ==================================================

        hod_intent = contains_any(
            query_normalized,
            [
                "hod",
                "head of the department",
                "head of department",
                "professor and head",
                "professor & head",
                "who is listed as professor and head",
            ],
        )

        programme_intent = contains_any(
            query_normalized,
            [
                "programme",
                "program",
                "programmes",
                "programs",
                "offered",
            ],
        )

        postgraduate_intent = contains_any(
            query_normalized,
            [
                "postgraduate",
                "post graduate",
                "pg",
            ],
        )

        undergraduate_intent = contains_any(
            query_normalized,
            [
                "undergraduate",
                "under graduate",
                "ug",
            ],
        )

        cutoff_intent = contains_any(
            query_normalized,
            [
                "cutoff",
                "cut-off",
                "opening cutoff",
                "closing cutoff",
                "opening cut-off",
                "closing cut-off",
            ],
        )

        minor_intent = contains_any(
            query_normalized,
            [
                "minor",
                "minor degree",
                "minor programme",
                "minor program",
            ],
        )

        erp_intent = contains_any(
            query_normalized,
            [
                "erp",
                "enterprise resource planning",
                "erp software",
                "erp system",
                "erp modules",
                "attendance module",
                "library module",
                "transport module",
                "examination module",
                "hostel module",
            ],
        )

        fee_portal_intent = contains_any(
            query_normalized,
            [
                "unified portal",
                "rec unified portal",
                "fee certificate",
                "fee certificates",
                "generate fee certificate",
                "generate fee certificates",
            ],
        )

        athletics_intent = contains_any(
            query_normalized,
            [
                "athletics",
                "athletic",
                "sports",
                "sports section",
                "student-life section",
                "student life section",
            ],
        )

        campus_location_intent = contains_any(
            query_normalized,
            [
                "campus location",
                "college campus",
                "where is the college",
                "where is rajalakshmi engineering college",
                "college located",
                "campus located",
                "college address",
                "campus address",
                "address of the college",
            ],
        )

        general_offer_intent = contains_any(
            query_normalized,
            [
                "general offer",
                "general offer salary",
                "general offer ctc",
                "general category",
            ],
        )

        super_dream_intent = contains_any(
            query_normalized,
            [
                "super dream",
                "super dream offer",
            ],
        )

        dream_offer_intent = contains_any(
            query_normalized,
            [
                "dream offer",
                "dream offer salary",
                "dream offer ctc",
            ],
        )

        # ==================================================
        # NEW: SPECIFIC PROGRAMME / PHD INTENT
        # ==================================================

        phd_intent = contains_any(
            query_normalized,
            [
                "ph.d",
                "phd",
                "doctor of philosophy",
            ],
        )

        placement_intent = contains_any(
            query_normalized,
            [
                "placement",
                "placements",
                "training and placement",
                "placement cell",
                "recruiter",
                "recruiters",
            ],
        )

        placement_purpose_intent = (
            placement_intent
            and contains_any(
                query_normalized,
                [
                    "purpose",
                    "role",
                    "function",
                    "what does",
                    "what is the purpose",
                ],
            )
        )

        placement_training_intent = (
            placement_intent
            and contains_any(
                query_normalized,
                [
                    "training",
                    "employability",
                    "value-added",
                    "value added",
                ],
            )
        )

        section_intent = contains_any(
            query_normalized,
            [
                "which section",
                "which student-life section",
                "which student life section",
                "which category",
                "which facility",
            ],
        )

        # ==================================================
        # 5. PROCESS CANDIDATES
        # ==================================================

        for index, document in enumerate(documents):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            text = document or ""

            text_normalized = " ".join(
                text.lower().split()
            )

            source_text = str(
                metadata.get(
                    "source",
                    "",
                )
            ).lower()

            file_text = str(
                metadata.get(
                    "file",
                    "",
                )
            ).lower()

            section_text = str(
                metadata.get(
                    "section",
                    "",
                )
            ).lower()

            combined_text = (
                source_text
                + " "
                + file_text
                + " "
                + section_text
                + " "
                + text_normalized
            )

            exact_score = 0.0

            # ==================================================
            # BASIC EXACT MATCHING
            # ==================================================

            if query_normalized in text_normalized:
                exact_score += 10.0

            if query_terms:

                matching_terms = sum(
                    1
                    for term in query_terms
                    if term in text_normalized
                )

                exact_score += (
                    matching_terms
                    / len(query_terms)
                ) * 3.0

            # ==================================================
            # FACTUAL KEYWORDS
            # ==================================================

            factual_keywords = [
                "credits",
                "prescribed",
                "degree",
                "programme",
                "program",
                "subject",
                "code",
                "placement",
                "placements",
                "recruiter",
                "recruiters",
                "offer",
                "offers",
                "dream",
                "super dream",
                "core",
                "salary",
                "ctc",
                "company",
                "companies",
                "duration",
                "cutoff",
                "cut-off",
            ]

            for keyword in factual_keywords:

                if (
                    keyword in query_normalized
                    and keyword in text_normalized
                ):
                    exact_score += 2.0

            # ==================================================
            # HOD
            # ==================================================

            if hod_intent and matched_department:

                aliases = department_aliases[
                    matched_department
                ]

                department_match = any(
                    alias in combined_text
                    for alias in aliases
                )

                if department_match:

                    exact_score += 35.0

                    if contains_any(
                        combined_text,
                        [
                            "hod",
                            "head of the department",
                            "head of department",
                            "professor & head",
                            "professor and head",
                        ],
                    ):
                        exact_score += 35.0

                    if "faculty" in combined_text:
                        exact_score += 12.0

                else:

                    exact_score -= 30.0

                    if contains_any(
                        combined_text,
                        [
                            "hod",
                            "head of the department",
                            "head of department",
                        ],
                    ):
                        exact_score -= 20.0

            # ==================================================
            # PROGRAMME / DEPARTMENT
            # ==================================================

            if programme_intent and matched_department:

                aliases = department_aliases[
                    matched_department
                ]

                if any(
                    alias in combined_text
                    for alias in aliases
                ):
                    exact_score += 25.0

                if contains_any(
                    combined_text,
                    [
                        "programmes offered",
                        "programme offered",
                        "program offered",
                        "undergraduate",
                        "postgraduate",
                        "b.e.",
                        "b.tech.",
                        "m.e.",
                        "m.tech.",
                        "ph.d",
                        "ph.d.",
                    ],
                ):
                    exact_score += 10.0

            # ==================================================
            # PHD + CSE
            # ==================================================

            if phd_intent:

                if "ph.d" in combined_text:
                    exact_score += 25.0

                if "phd" in combined_text:
                    exact_score += 20.0

                if (
                    matched_department
                    == "computer science and engineering"
                ):

                    if (
                        "computer science and engineering"
                        in combined_text
                    ):
                        exact_score += 40.0

                    if "ph.d" in combined_text:
                        exact_score += 25.0

                    if "phd" in combined_text:
                        exact_score += 20.0

                # Strongly reduce unrelated placement chunks.
                if (
                    "placement" in combined_text
                    and "ph.d" not in combined_text
                    and "phd" not in combined_text
                ):
                    exact_score -= 30.0

            # ==================================================
            # UG / PG
            # ==================================================

            if undergraduate_intent:

                if contains_any(
                    combined_text,
                    [
                        "undergraduate",
                        "ug",
                        "b.e.",
                        "b.tech.",
                    ],
                ):
                    exact_score += 10.0

                if contains_any(
                    combined_text,
                    [
                        "postgraduate",
                        "m.e.",
                        "m.tech.",
                        "mba",
                    ],
                ):
                    exact_score -= 8.0

            if postgraduate_intent:

                if contains_any(
                    combined_text,
                    [
                        "postgraduate",
                        "pg",
                        "m.e.",
                        "m.tech.",
                        "mba",
                    ],
                ):
                    exact_score += 10.0

                if contains_any(
                    combined_text,
                    [
                        "undergraduate",
                        "b.e.",
                        "b.tech.",
                    ],
                ):
                    exact_score -= 8.0

            # ==================================================
            # COURSE CODES
            # ==================================================

            course_codes = re.findall(
                r"\b[A-Z]{2,5}\d{4,6}\b",
                query.upper(),
            )

            for course_code in course_codes:

                if course_code.lower() in text_normalized:
                    exact_score += 25.0

            # ==================================================
            # COURSE NAMES
            # ==================================================

            course_keywords = [
                "theory of computation",
                "discrete mathematical structures",
                "problem solving and python programming",
                "digital logic and microprocessor",
                "data structures",
                "operating systems",
                "computer networks",
                "database management systems",
            ]

            for course_name in course_keywords:

                if course_name in query_normalized:

                    if course_name in text_normalized:
                        exact_score += 18.0

            # ==================================================
            # CUTOFF
            # ==================================================

            if cutoff_intent:

                if contains_any(
                    combined_text,
                    [
                        "cutoff table",
                        "cut-off table",
                        "cutoff",
                        "cut-off",
                        "opening rank",
                        "closing rank",
                        "opening cutoff",
                        "closing cutoff",
                        "admission cutoff",
                        "admission rank",
                    ],
                ):
                    exact_score += 40.0

                if matched_department:

                    aliases = department_aliases[
                        matched_department
                    ]

                    if any(
                        alias in combined_text
                        for alias in aliases
                    ):
                        exact_score += 25.0

                if contains_any(
                    combined_text,
                    [
                        "curriculum and syllabus",
                        "course outcomes",
                        "course code",
                    ],
                ) and not contains_any(
                    combined_text,
                    [
                        "cutoff",
                        "cut-off",
                        "opening rank",
                        "closing rank",
                    ],
                ):
                    exact_score -= 35.0

            # ==================================================
            # MINOR
            # ==================================================

            if minor_intent:

                if "minor degree" in combined_text:
                    exact_score += 35.0

                if "minor degree programmes" in combined_text:
                    exact_score += 30.0

                if "minor programmes" in combined_text:
                    exact_score += 25.0

                if "data science" in query_normalized:

                    if "data science" in combined_text:
                        exact_score += 20.0

                if (
                    "curriculum and syllabus"
                    in combined_text
                    and "minor"
                    not in combined_text
                ):
                    exact_score -= 30.0

            # ==================================================
            # COE
            # ==================================================

            coe_intent = contains_any(
                query_normalized,
                [
                    "controller of examinations",
                    "controller of examination",
                    "coe",
                ],
            )

            if coe_intent:

                if (
                    "controller of examinations"
                    in combined_text
                ):
                    exact_score += 35.0

                if "academic council" in combined_text:
                    exact_score += 15.0

            # ==================================================
            # PLACEMENT
            # ==================================================

            if placement_intent:

                if "training and placement cell" in combined_text:
                    exact_score += 30.0

                elif "placement" in combined_text:
                    exact_score += 8.0

                if "placement report" in combined_text:
                    exact_score += 15.0

                if "our recruiters" in combined_text:
                    exact_score += 12.0

                if "offer distribution" in combined_text:
                    exact_score += 10.0

                if "dream offer policy" in combined_text:
                    exact_score += 15.0

            # ==================================================
            # PLACEMENT PURPOSE
            # ==================================================

            if placement_purpose_intent:

                if "training and placement cell" in combined_text:
                    exact_score += 30.0

                if "training" in combined_text:
                    exact_score += 15.0

                if "employment" in combined_text:
                    exact_score += 20.0

                if "employability" in combined_text:
                    exact_score += 15.0

                if "purpose" in combined_text:
                    exact_score += 15.0

            # ==================================================
            # PLACEMENT TRAINING
            # ==================================================

            if placement_training_intent:

                if "training" in combined_text:
                    exact_score += 20.0

                if "employability" in combined_text:
                    exact_score += 20.0

                if "value-added" in combined_text:
                    exact_score += 20.0

                if "value added" in combined_text:
                    exact_score += 20.0

                if "training and placement cell" in combined_text:
                    exact_score += 20.0

            # ==================================================
            # GENERAL OFFER
            # ==================================================

            if general_offer_intent:

                if "general offer" in combined_text:
                    exact_score += 40.0

                if "lpa" in combined_text:
                    exact_score += 8.0

                if "ctc" in combined_text:
                    exact_score += 8.0

            # ==================================================
            # SUPER DREAM
            # ==================================================

            if super_dream_intent:

                if "super dream" in combined_text:
                    exact_score += 40.0

                if "lpa" in combined_text:
                    exact_score += 8.0

            # ==================================================
            # DREAM OFFER
            # ==================================================

            if dream_offer_intent:

                if "dream offer" in combined_text:
                    exact_score += 35.0

                if "dream offer policy" in combined_text:
                    exact_score += 20.0

                if "lpa" in combined_text:
                    exact_score += 8.0

            # ==================================================
            # LIBRARY
            # ==================================================

            library_terms = [
                "library",
                "central library",
                "books",
                "book",
                "e-resources",
                "e resources",
                "e-journals",
                "e journals",
                "e-books",
                "e books",
                "databases",
                "remote access",
                "digital resources",
                "digital library",
            ]

            if any(
                term in query_normalized
                for term in library_terms
            ):

                if "central library" in combined_text:
                    exact_score += 10.0

                elif "library" in combined_text:
                    exact_score += 5.0

                if "remote access" in combined_text:
                    exact_score += 10.0

                if "digital library" in combined_text:
                    exact_score += 15.0

                if "digital resources" in combined_text:
                    exact_score += 10.0

                if "e-resources" in combined_text:
                    exact_score += 8.0

            # ==================================================
            # ERP
            # ==================================================

            if erp_intent:

                if "erp" in combined_text:
                    exact_score += 40.0

                if (
                    "enterprise resource planning"
                    in combined_text
                ):
                    exact_score += 25.0

                if "erp" in source_text:
                    exact_score += 20.0

                for term in [
                    "attendance",
                    "library",
                    "transport",
                    "examination",
                    "hostel",
                ]:

                    if term in query_normalized:

                        if term in combined_text:
                            exact_score += 8.0

            # ==================================================
            # UNIFIED PORTAL / FEE
            # ==================================================

            if fee_portal_intent:

                if "unified portal" in combined_text:
                    exact_score += 45.0

                if "rec unified portal" in combined_text:
                    exact_score += 25.0

                if "fee certificate" in combined_text:
                    exact_score += 40.0

                if "fee certificates" in combined_text:
                    exact_score += 40.0

                if "fee" in combined_text:
                    exact_score += 8.0

            # ==================================================
            # ATHLETICS / SPORTS / STUDENT LIFE
            # ==================================================

            if athletics_intent:

                if "athletics" in combined_text:
                    exact_score += 40.0

                if "athletic" in combined_text:
                    exact_score += 25.0

                if "sports" in combined_text:
                    exact_score += 20.0

                if "sports complex" in combined_text:
                    exact_score += 20.0

                if "student life" in combined_text:
                    exact_score += 15.0

                if "technical societies" in query_normalized:

                    if "technical societies" in combined_text:
                        exact_score += 50.0

                if "health care" in query_normalized:

                    if "health care" in combined_text:
                        exact_score += 50.0

                if "dining" in query_normalized:

                    if "dining" in combined_text:
                        exact_score += 50.0

            # ==================================================
            # STUDENT-LIFE SECTION QUESTIONS
            # ==================================================

            if section_intent:

                section_names = [
                    "athletics",
                    "health care",
                    "technical societies",
                    "dining",
                    "living on campus",
                    "campus pulse",
                    "library",
                    "scholarships",
                    "awards",
                ]

                for section_name in section_names:

                    if section_name in query_normalized:

                        if section_name in combined_text:
                            exact_score += 50.0

                if "student life" in combined_text:
                    exact_score += 15.0

            # ==================================================
            # CAMPUS LOCATION
            # ==================================================

            if campus_location_intent:

                if "thandalam" in combined_text:
                    exact_score += 30.0

                if "602105" in combined_text:
                    exact_score += 30.0

                if "chennai" in combined_text:
                    exact_score += 15.0

                if "rajalakshmi engineering college" in combined_text:
                    exact_score += 10.0

            # ==================================================
            # Store result
            # ==================================================

            retrieved.append(
                {
                    "text": text,
                    "metadata": metadata or {},
                    "distance": distance,
                    "exact_score": exact_score,
                }
            )

        # ==================================================
        # 6. FINAL RANKING
        # ==================================================

        def ranking_score(
            item: dict[str, Any],
        ) -> float:

            distance = item.get(
                "distance"
            )

            distance_score = (
                -float(distance)
                if distance is not None
                else 0.0
            )

            return (
                float(
                    item.get(
                        "exact_score",
                        0.0,
                    )
                )
                + distance_score
            )

        retrieved.sort(
            key=ranking_score,
            reverse=True,
        )

        return retrieved[:k]

    def search_multiple(
        self,
        queries: list[str],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:

        all_results: list[dict[str, Any]] = []

        seen: set[tuple] = set()

        for query in queries:

            if not query.strip():
                continue

            results = self.search(
                query=query,
                top_k=top_k,
            )

            for result in results:

                metadata = (
                    result.get("metadata")
                    or {}
                )

                key = (
                    metadata.get("file"),
                    metadata.get("page"),
                    metadata.get("chunk"),
                )

                if key in seen:
                    continue

                seen.add(key)

                result["search_query"] = query

                all_results.append(result)

        return all_results