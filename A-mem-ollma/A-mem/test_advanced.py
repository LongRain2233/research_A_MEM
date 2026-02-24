from memory_layer import LLMController, AgenticMemorySystem
import os
import json
import argparse
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from openai import OpenAI
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from utils import calculate_metrics, aggregate_metrics
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

class advancedMemAgent:
    def __init__(self, model, backend, retrieve_k, temperature_c5, sglang_host="http://localhost", sglang_port=30000):
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port
        )
        self.retriever_llm = LLMController(
            backend=backend, 
            model=model, 
            api_key=None, 
            sglang_host=sglang_host, 
            sglang_port=sglang_port
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)
    
    def retrieve_memory_llm(self, memories_text, query):
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

                Conversation memories:
                {memories_text}

                Question: {query}

                Return only the relevant parts of the conversation that would help answer this specific question. Format your response as a JSON object with a "relevant_parts" field containing the selected text. 
                If no parts are relevant, do not do any things just return the input.

                Example response format:
                {{"relevant_parts": "2024-01-01: Speaker A said something relevant..."}}"""
            
            # Get LLM response
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "relevant_parts": {
                                        "type": "string",
                                    }
                                },
                                "required": ["relevant_parts"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        # print("response:{}".format(response))
        return response
    
    def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text. 

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""
            
            # Get LLM response
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "string",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        print("response:{}".format(response))
        try:
            response = json.loads(response)["keywords"]
        except:
            response = response.strip()
        return response

    def answer_question(self, question: str, category: int, answer: str) -> str:
        """Generate answer for a question given the conversation context."""
        keywords = self.generate_query_llm(question)
        # if category == 3:
        #     raw_context = self.retrieve_memory(keywords,k=10)
        #     # context = self.retrieve_memory_llm(raw_context, keywords)
        # else:
        raw_context = self.retrieve_memory(keywords,k=self.retrieve_k)
        context = raw_context
        # print("context:", context)
        # context = self.retrieve_memory_llm(raw_context, question)
        # context = raw_context
        assert category in [1,2,3,4,5]
        user_prompt = f"""Context:
                {context}

                Question: {question}

                Answer the question based only on the information provided in the context above."""
        temperature = 0.7
        if category == 5: # adversial question, follow the initial paper.
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""
                            Based on the context: {context}, answer the following question. {question} 
                            
                            Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:
                            """
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""
                            Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
                            Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.   

                            Question: {question} Short answer:
                            """
        elif category == 3:
            user_prompt = f"""
                            Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        response = self.memory_system.llm_controller.llm.get_completion(
            user_prompt,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                }
                            },
                            "required": ["answer"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }},temperature=temperature
        )
        # print(response)
        return response,user_prompt,raw_context

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('locomo_eval')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None, ratio: float = 1.0, backend: str = "sglang", temperature_c5: float = 0.5, retrieve_k: int = 10, sglang_host: str = "http://localhost", sglang_port: int = 30000):
    """Evaluate the agent on the LoComo dataset with checkpoint/resume support.
    
    Args:
        dataset_path: Path to the dataset file
        model: Name of the model to use
        output_path: Path to save results
        ratio: Ratio of dataset to evaluate
    """
    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    safe_model = model.replace(":", "_").replace("/", "_")
    log_filename = f"eval_ours_{safe_model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = setup_logger(log_path)
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Load dataset
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Select subset of samples based on ratio
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")
    
    # ========== Checkpoint / Resume Support ==========
    memories_dir = os.path.join(os.path.dirname(__file__), "cached_memories_advanced_{}_{}".format(backend, safe_model))
    os.makedirs(memories_dir, exist_ok=True)
    
    # Checkpoint file for QA evaluation progress
    checkpoint_file = os.path.join(memories_dir, "eval_checkpoint.json")
    
    # Try to load existing checkpoint
    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    error_num = 0
    completed_samples = set()  # Track which samples have finished QA evaluation
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            results = checkpoint.get("results", [])
            all_metrics = checkpoint.get("all_metrics", [])
            all_categories = checkpoint.get("all_categories", [])
            total_questions = checkpoint.get("total_questions", 0)
            category_counts = defaultdict(int, {int(k): v for k, v in checkpoint.get("category_counts", {}).items()})
            error_num = checkpoint.get("error_num", 0)
            completed_samples = set(checkpoint.get("completed_samples", []))
            logger.info(f"[Checkpoint] Resumed from checkpoint: {len(completed_samples)} samples completed, {total_questions} questions answered")
        except Exception as e:
            logger.info(f"[Checkpoint] Failed to load checkpoint ({e}), starting fresh")
    
    def save_checkpoint():
        """Save current evaluation progress to checkpoint file."""
        checkpoint = {
            "results": results,
            "all_metrics": all_metrics,
            "all_categories": all_categories,
            "total_questions": total_questions,
            "category_counts": dict(category_counts),
            "error_num": error_num,
            "completed_samples": list(completed_samples)
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
    
    # ========== Main Evaluation Loop ==========
    allow_categories = [1,2,3,4,5]
    for sample_idx, sample in enumerate(samples):
        
        # Skip already completed samples
        if sample_idx in completed_samples:
            logger.info(f"[Checkpoint] Skipping sample {sample_idx} (already completed)")
            continue
        
        agent = advancedMemAgent(model, backend, retrieve_k, temperature_c5, sglang_host, sglang_port)
        
        # Create memory cache filename based on sample and session indices
        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy")
        # Incremental memory build progress file
        memory_progress_file = os.path.join(memories_dir, f"memory_progress_sample_{sample_idx}.json")

        # ---- Phase 1: Build or Load Memories ----
        if os.path.exists(memory_cache_file):
            # Full memory cache exists, load it
            logger.info(f"Loading cached memories for sample {sample_idx}")
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            if os.path.exists(retriever_cache_file):
                print(f"Found retriever cache files:")
                print(f"  - Retriever cache: {retriever_cache_file}")
                print(f"  - Embeddings cache: {retriever_cache_embeddings_file}")
                agent.memory_system.retriever = agent.memory_system.retriever.load(retriever_cache_file, retriever_cache_embeddings_file)
            else:
                print(f"No retriever cache found at {retriever_cache_file}, loading from memory")
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(cached_memories, 'all-MiniLM-L6-v2')
            print(agent.memory_system.retriever.corpus)
            logger.info(f"Successfully loaded {len(cached_memories)} memories")
        else:
            # Need to build memories - with incremental saving
            logger.info(f"Building memories for sample {sample_idx}...")
            
            # Flatten all turns with their timestamps for indexing
            all_turns = []
            for _, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    all_turns.append((turns.date_time, turn))
            
            # Check if there's a partial progress to resume from
            start_turn_idx = 0
            partial_memory_file = os.path.join(memories_dir, f"memory_partial_sample_{sample_idx}.pkl")
            partial_retriever_file = os.path.join(memories_dir, f"retriever_partial_sample_{sample_idx}.pkl")
            partial_embeddings_file = os.path.join(memories_dir, f"embeddings_partial_sample_{sample_idx}.npy")
            
            if os.path.exists(memory_progress_file):
                try:
                    with open(memory_progress_file, 'r') as f:
                        progress = json.load(f)
                    start_turn_idx = progress.get("next_turn_idx", 0)
                    if start_turn_idx > 0 and os.path.exists(partial_memory_file):
                        with open(partial_memory_file, 'rb') as f:
                            agent.memory_system.memories = pickle.load(f)
                        if os.path.exists(partial_retriever_file):
                            agent.memory_system.retriever = agent.memory_system.retriever.load(partial_retriever_file, partial_embeddings_file)
                        else:
                            agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(agent.memory_system.memories, 'all-MiniLM-L6-v2')
                        logger.info(f"[Checkpoint] Resuming memory building from turn {start_turn_idx}/{len(all_turns)} ({len(agent.memory_system.memories)} memories loaded)")
                except Exception as e:
                    logger.info(f"[Checkpoint] Failed to load partial progress ({e}), starting from scratch")
                    start_turn_idx = 0
            
            # Build memories with periodic saving
            save_interval = 20  # Save every 20 turns
            for turn_idx in range(start_turn_idx, len(all_turns)):
                turn_datetime, turn = all_turns[turn_idx]
                conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                agent.add_memory(conversation_tmp, time=turn_datetime)
                
                # Periodic incremental save
                if (turn_idx + 1) % save_interval == 0:
                    logger.info(f"  [Memory] Turn {turn_idx + 1}/{len(all_turns)} processed, saving checkpoint...")
                    with open(partial_memory_file, 'wb') as f:
                        pickle.dump(agent.memory_system.memories, f)
                    agent.memory_system.retriever.save(partial_retriever_file, partial_embeddings_file)
                    with open(memory_progress_file, 'w') as f:
                        json.dump({"next_turn_idx": turn_idx + 1}, f)
            
            # All turns done - save final memory cache
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(agent.memory_system.memories, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            logger.info(f"Successfully cached {len(agent.memory_system.memories)} memories")
            
            # Clean up partial files
            for f_path in [partial_memory_file, partial_retriever_file, partial_embeddings_file, memory_progress_file]:
                if os.path.exists(f_path):
                    os.remove(f_path)
        
        # ---- Phase 2: QA Evaluation ----
        logger.info(f"\nProcessing QA for sample {sample_idx + 1}/{len(samples)}")
        
        for qa in sample.qa:
            if int(qa.category) in allow_categories:
                total_questions += 1
                category_counts[qa.category] += 1
                
                # Generate prediction
                prediction, user_prompt, raw_context = agent.answer_question(qa.question, qa.category, qa.final_answer)
                try:
                    parsed = json.loads(prediction)
                    prediction = parsed.get("short_answer") or parsed.get("answer") or prediction
                except:
                    prediction = prediction
                    logger.info(f"Failed to parse prediction as JSON: {prediction}")
                    error_num += 1
                # Log results
                logger.info(f"\nQuestion {total_questions}: {qa.question}")
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Reference: {qa.final_answer}")
                logger.info(f"User Prompt: {user_prompt}")
                logger.info(f"Category: {qa.category}")
                logger.info(f"Raw Context: {raw_context}")
                
                # Calculate metrics
                metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                    "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0, 
                    "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, 
                    "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
                }
                
                all_metrics.append(metrics)
                all_categories.append(qa.category)
                
                # Store individual result
                result = {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": qa.category,
                    "metrics": metrics
                }
                results.append(result)
                
                # Log progress
                if total_questions % 10 == 0:
                    logger.info(f"Processed {total_questions} questions")
        
        # ---- Mark sample as completed and save checkpoint ----
        completed_samples.add(sample_idx)
        save_checkpoint()
        logger.info(f"[Checkpoint] Sample {sample_idx} completed and saved ({total_questions} total questions so far)")
    
    # ========== Final Results ==========
    # Calculate aggregate metrics
    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    
    # Prepare final results
    final_results = {
        "model": model,
        "dataset": dataset_path,
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "individual_results": results
    }
    logger.info(f"Error number: {error_num}")
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    # Log summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Total questions evaluated: {total_questions}")
    if total_questions > 0:
        logger.info("\nCategory Distribution:")
        for category, count in sorted(category_counts.items()):
            logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")
        
        logger.info("\nAggregate Metrics:")
        for split_name, metrics in aggregate_results.items():
            logger.info(f"\n{split_name.replace('_', ' ').title()}:")
            for metric_name, stats in metrics.items():
                logger.info(f"  {metric_name}:")
                for stat_name, value in stats.items():
                    logger.info(f"    {stat_name}: {value:.4f}")
    
    # Clean up checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("[Checkpoint] All samples completed, checkpoint file removed")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate text-only agent on LoComo dataset")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                      help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                      help="OpenAI model to use")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                      help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="sglang",
                      help="Backend to use (openai, ollama, or sglang)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                      help="Temperature for the model")
    parser.add_argument("--retrieve_k", type=int, default=10,
                      help="Retrieve k")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                      help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                      help="SGLang server port (for sglang backend)")
    args = parser.parse_args()
    
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    
    # Convert relative path to absolute path
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    if args.output:
        output_path = os.path.join(os.path.dirname(__file__), args.output)
    else:
        output_path = None
    
    evaluate_dataset(dataset_path, args.model, output_path, args.ratio, args.backend, args.temperature_c5, args.retrieve_k, args.sglang_host, args.sglang_port)

if __name__ == "__main__":
    main()