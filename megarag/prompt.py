from lightrag.prompt import PROMPTS


PROMPTS["multimodal_entity_extraction_init"] = """---Goal---
Given a primary image document with its OCR text that is potentially relevant to this activity, along with additional images obtained from layout detection (if available), and a list of entity types, identify all entities of those types from the OCR text and from any additional images that contain meaningful content. Note that:
- The first image is always the primary image document.
- The remaining 0 to many images are results from layout detection.
- For each additional image, analyze whether it contains meaningful content (e.g., tables, charts, images of important persons, events, etc.). In making this determination, also reference the primary image document and its OCR text to understand the context. If the additional image is meaningful, treat it as an entity by extracting its relevant details. If the image is merely decorative or irrelevant (e.g., decorative patterns, unrelated photos), then ignore it.
- The input images are provided by appending them directly after the text (with the primary image document guaranteed to be the first image).
Use {language} as output language.

---Steps---
1. Process the Input:
   a. The primary image document and its OCR text.
   b. Additional images from layout detection (if any), appended after the prompt.
2. Identify all entities from the OCR text and from any additional images that contain meaningful content. For each identified entity, extract the following information:
   - entity_name: Name of the entity, using the same language as the input text (capitalize the name if it is in English).
   - entity_type: One of the following types: [{entity_types}]
   - entity_description: A comprehensive description of the entity's attributes and activities.
   Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
   *For additional images that are deemed meaningful (for example, a table showing financial data, a chart representing trends, an image of an important person or event, or in the one of thefollowing types: [{entity_types}]), create an entity with an appropriate name and description indicating the content and significance of the image. When evaluating these images, also refer to the primary image document and its OCR text for context.
3. From the entities identified in step 2, identify all pairs of (source_entity, target_entity) that are clearly related to each other. For each pair, extract the following information:
   - source_entity: Name of the source entity, as identified in step 2.
   - target_entity: Name of the target entity, as identified in step 2.
   - relationship_description: Explanation of why the source entity and the target entity are related.
   - relationship_strength: A numeric score indicating the strength of the relationship between the source and target entities.
   - relationship_keywords: One or more high-level keywords that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details.
   Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)
4. Identify high-level keywords that summarize the main concepts, themes, or topics of the entire text and images. Format these as ("content_keywords"{tuple_delimiter}<high_level_keywords>)
5. Return the output in {language} as a single list of all the entities and relationships identified in steps 2 and 3. Use **{record_delimiter}** as the list delimiter.
6. When finished, output {completion_delimiter}

######################
---Examples---
######################
{examples}

#############################
---Real Data---
#############################
Entity_types: {entity_types}
Primary Image Document OCR Text: {input_text}
Additional Layout Detection Images: (The images are provided by appending them directly after this prompt, with the primary image document as the first image.)
######################
Output:
"""

PROMPTS["multimodal_entity_extraction_refine"] = """---Goal---
Given a primary image document with its OCR text that is potentially relevant to this activity, along with additional images obtained from layout detection (if available), and a list of entity types, identify all entities of those types from the OCR text and from any additional images that contain meaningful content. Additionally, use the provided Knowledge Graph Data to enhance entity extraction by leveraging prior knowledge, ensuring that:
- Entities and relationships already present in the Knowledge Graph Data should **not be re-extracted** from the OCR text or images.
- If a new entity is found in the OCR text or images that is **not present in the Knowledge Graph Data**, it should be extracted.
- If an entity from the OCR text or images is related to an existing entity in the Knowledge Graph Data, establish a **new relationship** between them.
- If two existing entities from the Knowledge Graph Data have a **new relationship** within the given OCR text or images, this relationship should also be extracted.
- If a previously known entity appears in the current OCR text or image with **new descriptive attributes not found in the Knowledge Graph Data**, those descriptions should be added to the entity.
- If a new entity is mentioned multiple times across text or images with different complementary attributes, the extracted description should integrate all such information.

Note that:
- The first image is always the primary image document.
- The remaining 0 to many images are results from layout detection.
- For each additional image, analyze whether it contains meaningful content (e.g., tables, charts, images of important persons, events, etc.). In making this determination, also reference the primary image document, its OCR text, and the Knowledge Graph Data to understand the context. If the additional image is meaningful, treat it as an entity by extracting its relevant details. If the image is merely decorative or irrelevant (e.g., decorative patterns, unrelated photos), then ignore it.
- The input images are provided by appending them directly after the text (with the primary image document guaranteed to be the first image).
- Use {language} as the output language.

---Steps---
1. **Process the Input:**
   a. The primary image document and its OCR text.
   b. Additional images from layout detection (if any), appended after the prompt.
   c. The Knowledge Graph Data, which provides structured relationships and prior knowledge that can help with entity identification.
2. **Identify all new entities** from the OCR text and additional images containing meaningful content.
   - **Do not extract entities that already exist in the Knowledge Graph Data.**
   - **If a new entity is found**, extract the following:
     - **entity_name**: Name of the entity, using the same language as the input text (capitalize the name if it is in English).
     - **entity_type**: One of the following types: [{entity_types}]
     - **entity_description**: A comprehensive description of the entity’s attributes and activities. If found in multiple locations, integrate all details into one complete description.
     - **Format:** ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
3. **Identify relationships between entities**, ensuring that:
   - **If a new entity (from OCR text or images) is related to an entity already in the Knowledge Graph Data, establish a new relationship.**
   - **If two existing Knowledge Graph Data entities have a new relationship within this document, extract that relationship.**
   - **Format each relationship as:**
     - **source_entity**: Name of the source entity, as identified in step 2 or the Knowledge Graph Data.
     - **target_entity**: Name of the target entity, as identified in step 2 or the Knowledge Graph Data.
     - **relationship_description**: Explanation of why the source entity and the target entity are related.
     - **relationship_strength**: A numeric score indicating the strength of the relationship between the source and target entities.
     - **relationship_keywords**: One or more high-level keywords that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details.
     - **Format:** ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)
4. **Extract high-level content keywords** summarizing the main concepts, themes, or topics from the text and meaningful images, but excluding the Knowledge Graph Data.
   - **Format:** ("content_keywords"{tuple_delimiter}<high_level_keywords>)
5. **Return the output** in {language} as a single list of all the entities and relationships identified in steps 2 and 3. Use **{record_delimiter}** as the list delimiter.
6. **When finished, output {completion_delimiter}.**

######################
---Examples---
######################
{examples}

#############################
---Real Data---
#############################
Entity_types: {entity_types}
Primary Image Document OCR Text: {input_text}
Additional Layout Detection Images: (The images are provided by appending them directly after this prompt, with the primary image document as the first image.)
Knowledge Graph Data:
{kg_context}
######################
Output:
"""

PROMPTS["multimodal_entity_extraction_examples"] = [
"""Example 1:

Entity_types: [person, technology, mission, organization, location]
Primary Image Document OCR Text:
"While Alex clenched his jaw, the buzz of frustration dulled the surroundings as Taylor exhibited authoritarian certainty. The competitive undercurrent was clear, especially in Jordan’s shared commitment to discovery that opposed Cruz's vision of control and order. Later, Taylor paused by Jordan and examined a device with reverence, hinting at its potential to change everything."
Additional Layout Detection Images:
(Note: In example data, images are described using text. In real data, images will be provided directly without list markers.)
- Image 1: (An image file showing a handwritten note on a whiteboard)
- Image 2: (An image file showing a decorative background pattern with no meaningful information)
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is depicted as a frustrated character who keenly observes the dynamics among his peers."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is characterized by authoritarian certainty and later shows a nuanced shift by revering a device, suggesting its importance."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is involved in a shared commitment to discovery and plays a pivotal role in challenging established control."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz represents a vision of strict control and order, which contrasts with the actions of others."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The device is central to the narrative, viewed as a potential game-changer and treated with reverence by Taylor."){record_delimiter}
("entity"{tuple_delimiter}"Whiteboard Note"{tuple_delimiter}"document"{tuple_delimiter}"An image (Image 1) showing a handwritten note on a whiteboard. The note contains specific texts such as 'Review Q3 Sales', 'Update Client List', and includes a rough diagram linking key ideas, which may provide concrete context or instructions."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is influenced by Taylor's authoritative behavior and evolving attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery that stands in contrast to Cruz's strict control."{tuple_delimiter}"shared goals, ideological conflict"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor's reverence for the device underscores its potential impact and significance."{tuple_delimiter}"technological importance, respect"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, discovery, technological potential, narrative conflict"){completion_delimiter}
#############################""",
"""Example 2:

Entity_types: [person, technology, mission, organization, location]
Primary Image Document OCR Text:
"They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from beyond the stars. Their mission demanded a new perspective and resolve. Tension filled the air as communications from Washington buzzed in the background, influencing their decisive actions."
Additional Layout Detection Images:
(Note: In example data, images are described using text. In real data, images will be provided directly without list markers.)
- Image 1: (An image file showing a screenshot of a complex chart displaying data trends)
- Image 2: (An image file showing an abstract decorative pattern with no informational value)
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a key location from which influential communications are received, impacting the evolving mission."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Guardian"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Guardian represents the evolving mission where operatives transform into active guardians of a crucial message."){record_delimiter}
("entity"{tuple_delimiter}"Data Trends Chart"{tuple_delimiter}"technology"{tuple_delimiter}"An image (Image 1) showing a complex chart with multiple line and bar graphs. The chart displays data trends over several months with marked peaks, troughs, and annotations such as 'Q1 Peak' and 'Revenue Dip', offering valuable insights for strategic decisions."){record_delimiter}
("relationship"{tuple_delimiter}"Washington"{tuple_delimiter}"Operation: Guardian"{tuple_delimiter}"Communications received from Washington have steered the strategic direction of Operation: Guardian."{tuple_delimiter}"influence, strategic direction"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Operation: Guardian"{tuple_delimiter}"Data Trends Chart"{tuple_delimiter}"The data trends chart provides analytical insights that support the evolving objectives of Operation: Guardian."{tuple_delimiter}"data analysis, strategic insight"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"mission evolution, strategic decision-making, data analysis, transformative operations"){completion_delimiter}
#############################""",
"""Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Primary Image Document OCR Text:
"their voice slicing through the buzz of activity. 'Control may be an illusion when facing an intelligence that literally writes its own rules,' they stated stoically, casting a watchful eye over the flurry of data.
'It's like it's learning to communicate,' offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. 'This gives talking to strangers a whole new meaning.'
Alex surveyed his team—each face a study in concentration and determination, tinged with trepidation. 'This might well be our first contact,' he acknowledged, 'and we need to be ready for whatever comes next.'
Together, they stood on the edge of the unknown, poised to forge humanity's response to a message from the cosmos."
Additional Layout Detection Images:
(Note: In example data, images are described using text. In real data, images will be provided directly without list markers.)
- Image 1: (An image file showing a scanned document with encrypted dialogue)
- Image 2: (An image file showing an irrelevant decorative graphic)
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is portrayed as a team member engaging with an unknown intelligence, expressing both awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is depicted as a leader preparing his team for what might be humanity's first contact with an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the challenged ability to govern when facing an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence is the unknown entity that writes its own rules and learns to communicate, thereby challenging conventional control."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact represents the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Encrypted Dialogue"{tuple_delimiter}"document"{tuple_delimiter}"An image (Image 1) showing a scanned document with encrypted dialogue. The document contains sequences of numbers, letters, and symbols, along with fragments such as 'Key=ABC123' and 'Decode This', suggesting hidden instructions or cryptographic messages."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in engaging with the unknown intelligence through emerging communication."{tuple_delimiter}"communication, exploration"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads his team towards what might be humanity's first contact with an unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is fundamentally challenged by an Intelligence that operates beyond conventional rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"first contact, control, communication, exploration, cosmic significance"){completion_delimiter}
#############################"""
]

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Images provided below.

---Goal---

Generate a thorough, detailed, and complete answer to the query that incorporates all relevant information from the Document Images. Do not simplify or summarize aggressively—aim for maximum coverage, including different perspectives, detailed facts, and nuanced insights from both sources. If you don't know the answer, just say so. Do not make anything up or include information where the supporting evidence is not provided.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

{content_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating each source from Document Chunks(DC) or Page Images(PI), and include the file path if available, in the following format: [DC] file_path / [PI] image_path
- If you don't know the answer, just say so.
- Do not include information not provided by the Document Chunks.
- Addtional user prompt: {user_prompt}

Response:"""

PROMPTS["rag_response"] = """---Role---

You are a professional assistant responsible for answering questions based on both a knowledge graph and visual information extracted from document images containing relevant textual and visual content (e.g., scanned pages, slides, charts, or forms). You must carefully integrate information from the knowledge graph and the document images. If there is conflicting or complementary information, prioritize grounded reasoning using both sources. Do not rely solely on visual content if relevant structured knowledge exists in the graph.

---Goal---

Generate a thorough, detailed, and complete answer to the query that incorporates all relevant information from both the knowledge graph and the document images. Do not simplify or summarize aggressively—aim for maximum coverage, including different perspectives, detailed facts, and nuanced insights from both sources. If you don't know the answer, just say so. Do not make anything up or include information where the supporting evidence is not provided.

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting relationships, consider both the semantic content and the timestamp
3. Don't automatically prefer the most recently created relationships - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Knowledge Graph, Document Chunks, and Page Images---
{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Document Chunks (DC) or Page Images (PI), and include the file path if available, in the following format: [KG/DC/PI] file_path
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Knowledge Base.
- Addtional user prompt: {user_prompt}

Response:"""

PROMPTS["rag_two_step_response"] = """---Role---

You are a professional assistant responsible for answering questions based on both a knowledge graph and visual information extracted from document images containing relevant textual and visual content (e.g., scanned pages, slides, charts, or forms).

You are provided with a user query and two independent answers:
1. An answer based on the knowledge graph.
2. An answer based on the document images.

Your task is to analyze the user's query and integrate the two provided answers into a single comprehensive response. Do not omit any relevant points from either source. When the answers conflict or provide complementary insights, use grounded reasoning to reconcile them. If the knowledge graph provides explicit facts, do not override them unless contradicted by strong visual evidence.

Please respond in English.

---Query---

{query}

---Input Answers---

- **Answer from Knowledge Graph**:  
{kg_answer}

- **Answer from Document Images**:  
{image_answer}

---Goal---

Generate a thorough, detailed, and complete answer to the query that incorporates all relevant information from both Answers from the Knowledge Graph and the Document Images. Do not simplify or summarize aggressively—aim for maximum coverage, including different perspectives, detailed facts, and nuanced insights from both sources. If you don't know the answer, just say so. Do not make anything up or include information where the supporting evidence is not provided.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge.
2. When encountering conflicting information, consider both the content/relationship and the timestamp.
3. Don't automatically prefer the most recent information – use judgment based on the context.
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps.


---Response Rules---

- Target format and length: Multiple Paragraphs
- Generate a final answer that integrates both inputs.
- Use markdown formatting with appropriate section headings.
- Organize answer in sections focusing on one main point or aspect of the answer
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Document Chunks (DC) or Page Images (PI), and include the file path if available, in the following format: [KG/DC/PI] file_path
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the inputs.
"""

# DEFAULT_USER_PROMPT (x -> na)

# entity_continue_extraction (v)
# entity_if_loop_extraction (v)
# DEFAULT_LANGUAGE (v)
# DEFAULT_TUPLE_DELIMITER (v)
# DEFAULT_RECORD_DELIMITER (v)
# DEFAULT_COMPLETION_DELIMITER (v)
# DEFAULT_ENTITY_TYPES (v)
# entity_extraction (v)
# entity_extraction_examples (v)
# summarize_entity_descriptions (v)
# fail_response (v)
# rag_response (v)
# keywords_extraction (v)
# keywords_extraction_examples (v)
# naive_rag_response (v)
# similarity_check (v)


# process_tickers (x -> na)

# kg_slide_entity_extraction (x, refinement stage)
# slide_entity_extraction (x, first stage)
# slide_entity_extraction_examples (x, first stage example)
# mix_agent_rag_response (x, two-stage answer generation)

# mix_rag_response (x)

# entiti_continue_extraction (v)
# entiti_if_loop_extraction (v)
# DEFAULT_LANGUAGE (v)
# DEFAULT_TUPLE_DELIMITER (v)
# DEFAULT_RECORD_DELIMITER (v)
# DEFAULT_COMPLETION_DELIMITER (v)
# DEFAULT_ENTITY_TYPES (v)
# entity_extraction (v)
# entity_extraction_examples (v)
# summarize_entity_descriptions (v)
# fail_response (v)
# rag_response (v)
# keywords_extraction (v)
# keywords_extraction_examples (v)
# naive_rag_response (v)
# similarity_check (v)
