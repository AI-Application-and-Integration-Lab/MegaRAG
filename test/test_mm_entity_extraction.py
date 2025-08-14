import json
import random
import asyncio
import logging

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from lightrag.utils import TokenTracker
from megarag.operate import extract_entities
from megarag.llms.openai import gpt_4o_mini_complete 

logger = logging.getLogger("extract_test")
logging.basicConfig(level=logging.INFO, format="%(message)s")

TextChunkSchema = Dict[str, Any]
def build_fake_input() -> Tuple[Dict[str, TextChunkSchema], Dict[str, Any]]:
    chunks = {
        "chunk-ed36296e0561195ce6e39310fc98a0a1": {
            "tokens": 550,
            # "content": "Section Summaries\nSection summaries distill the information in each section for both students and instructors down to key, concise points addressed in the section.\nKey Terms\nKey terms are bold and are followed by a definition in context. Definitions of key terms are also listed in each end-of-chapter Glossary, as well as a book-level Glossary appendix.\nAssessments\nA variety of assessments allow instructors to confirm core conceptual understanding, elicit brief explanations that demonstrate student understanding, and offer more in-depth assignments that enable learners to dive more deeply into a topic or history-study skill.\nReview Questions test for conceptual apprehension of key concepts.   \n• Check Your Understanding Questions require students to explain concepts in words.   \n• Application and Reflection Questions dive deeply into the material to support longer reflection, group discussion, or written assignments.\nAnswers to Questions in the Book\nThe end-of-chapter Review, Check Your Understanding, and Reflection Questions are intended for homework assignments or classroom discussion; thus, student-facing answers are not provided in the book. Answers and sample answers are provided in the Instructor Answer Guide, for instructors to share with students at their discretion, as is standard for such resources.\nAbout the Authors\nSenior Contributing Authors\nSenior Contributing Authors (left to right): Ann Kordas, Ryan J. Lynch, Brooke Nelson, Julie Tatlock.\nAnn Kordas, Johnson & Wales University\nAnn Kordas holds a PhD in History from Temple University, and a JD from Boston University School of Law. She is a professor in the Humanities Department at Johnson & Wales University, where she teaches courses in U.S. history, world history, the history of the Atlantic World, and the history of the Pacific World. Her research interests lie primarily in the fields of cultural history and gender history.\nRyan J. Lynch, Columbus State University\nDr. Ryan J. Lynch is Associate Professor of the History of the Islamic World and Associate Dean of the Honors College at Columbus State University in Columbus, Georgia. A specialist of pre-modern Islamic history, he completed his DPhil and MPhil in Islamic Studies and History at the University of Oxford, an MLitt in Middle Eastern History and Language at the University of St Andrews, and a BA in History and Religious Studies at Stetson University. Dr. Lynch’s research focuses primarily on the period of the early Islamic conquests, the Islamization of the Middle East, Islamic state formation, and Arabic historiography, while he also has a growing interest in how modern terror groups use an imagined Islamic past to justify their extremist views in the modern period. He is the author of the award-winning book ArabConquestsandEarlyIslamicHistoriography (I.B. Tauris, 2020).",
            "content": "Please see the figures.",
            "page_img": "dumps/World_History_Volume_1/auto/page_images/World_History_Volume_1_page_013.jpeg",
            "fig_imgs": [
                "dumps/World_History_Volume_1/auto/images/fe9219515c87ec17bd217339f90f10f2cfe76f2493c37226a583ed4429290683.jpg"
            ],
            "chunk_order_index": 12,
            "full_doc_id": "doc-7c99b85add5cb6f940c3611cad09df84",
            "file_path": "dumps/World_History_Volume_1/pages_content.json",
        }
    }
    token_tracker = TokenTracker()
    async def llm_func(
        prompt,
        input_images=None,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ):
        results = await gpt_4o_mini_complete(
            prompt=prompt,
            input_images=input_images,
            system_prompt=system_prompt,
            history_messages=history_messages,
            keyword_extraction=keyword_extraction,
            token_tracker=token_tracker,
            **kwargs,
        )
        return results

    global_config = {
        "llm_model_func": llm_func,
        "entity_extract_max_gleaning": 0,
        "addon_params": {
            "example_number": 3,
        },
        "llm_model_max_async": 2,
    }
    return chunks, global_config, token_tracker

def results_to_jsonable(results):
    jsonable = []
    for maybe_nodes, maybe_edges in results:
        # nodes: flatten the defaultdict[str, list[dict]]
        nodes_list = []
        for _name, entities in dict(maybe_nodes).items():
            nodes_list.extend(entities)

        # edges: turn tuple keys into explicit src/tgt fields
        edges_list = []
        for (src, tgt), rel_list in dict(maybe_edges).items():
            for rel in rel_list:
                edges_list.append({"src": src, "tgt": tgt, **rel})

        jsonable.append({"nodes": nodes_list, "edges": edges_list})
    return jsonable

async def main():
    chunks, cfg, token_tracker = build_fake_input()
    results = await extract_entities(
        chunks=chunks,
        global_config=cfg,
    )
    print("\n=== extract_entities() returned ===")
    safe_results = results_to_jsonable(results)
    print(json.dumps(safe_results, indent=4))
    # print(json.dumps(results, indent=4))
    # for idx, (nodes, edges) in enumerate(results, start=1):
    #     print(f"Chunk‑{idx}:")
    #     print("  Nodes:", list(nodes.keys()))
    #     print("  Edges:", list(edges.keys()))
    print(f'Total token usage: {token_tracker}')

if __name__ == "__main__":
    asyncio.run(main())
