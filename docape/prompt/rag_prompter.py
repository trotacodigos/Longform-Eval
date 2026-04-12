def build_prompt(entry: dict):
    system_prompt = (
        "You are a discourse analyst. Your task is to "
        "analyze a document and extract its key discourse "
        "attributes. Be concise and precise. "
        "Output ONLY valid JSON."
    )

    genre = entry["domain"]
    src_doc = entry["src_doc"]

    genre_examples = {
        "news": {
            "text_type": (
                "news report, investigative article, "
                "opinion piece, editorial, press release, "
                "interview, feature article"
            ),
            "domain": (
                "politics, diplomacy, economics, finance, "
                "science, technology, environment, health, "
                "sports, culture, society, crime, "
                "international affairs, military, education"
            ),
        },
        "social": {
            "text_type": (
                "social media post, blog post, online comment, "
                "forum thread, product review, newsletter, "
                "personal essay"
            ),
            "domain": (
                "lifestyle, travel, food, fashion, "
                "entertainment, gaming, parenting, "
                "health and wellness, personal finance, "
                "technology, politics, sports"
            ),
        },
        "literary": {
            "text_type": (
                "literary narrative, literary dialogue, "
                "short story, novel excerpt, poetry, drama, "
                "personal memoir, historical fiction"
            ),
            "domain": (
                "coming-of-age, romance, historical, thriller, "
                "family, war, social commentary, philosophical, "
                "fantasy, biographical"
            ),
        },
    }

    examples = genre_examples.get(genre, {
        "text_type": "other",
        "domain": "other"
    })

    user_prompt = (
        f'The following source document belongs to the '
        f'"{genre}" genre. Analyze its content and extract '
        f'the three discourse attributes listed below. '
        f'Base your analysis solely on the document content.\n\n'
        f'[Source Document]\n{src_doc}\n\n'
        f'Use the genre label to select the most appropriate '
        f'text_type and domain from the examples provided, '
        f'then extract all attributes and output a JSON object.\n\n'
        f'Genre: {genre}\n'
        f'  text_type examples: {examples["text_type"]}\n'
        f'  domain examples: {examples["domain"]}\n\n'
        'Output the following JSON structure:\n\n'
        '{\n'
        '  "genre_and_domain": {\n'
        '    "text_type": "<select from examples above>",\n'
        '    "domain": "<select from examples above>"\n'
        '  },\n'
        '  "participant_relationships": {\n'
        '    "identified": <true | false>,\n'
        '    "participants": [\n'
        '      {\n'
        '        "role": "<e.g. narrator, interviewer, '
        'character, customer, agent>",\n'
        '        "description": "<brief description '
        'if identifiable>"\n'
        '      }\n'
        '    ],\n'
        '    "relationship_type": "<e.g. '
        'interviewer--interviewee, narrator--character, '
        'customer--agent, none identifiable>"\n'
        '  },\n'
        '  "register_and_formality": {\n'
        '    "overall_register": "<one of: formal | '
        'semi-formal | informal | mixed>",\n'
        '    "korean_speech_level": "<one of: '
        'formal haepsyo-che | formal haeyo-che | '
        'informal banmal | mixed | not applicable>",\n'
        '    "chinese_register": "<one of: '
        'written | colloquial | mixed | not applicable>",\n'
        '    "notes": "<any additional register '
        'observations relevant to translation>"\n'
        '  }\n'
        '}\n\n'
        'Rules:\n'
        '- Use the provided genre label as the primary '
        'guide for text_type and domain selection.\n'
        '- You may use values outside the examples '
        'if none fits, but prefer the listed options.\n'
        '- If participant relationships cannot be '
        'identified, set "identified" to false and '
        'leave "participants" as an empty list.\n'
        '- Do not infer information not present '
        'in the document.\n'
        '- Output ONLY the JSON object, '
        'no preamble or explanation.'
    )

    return system_prompt, user_prompt