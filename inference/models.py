# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Models used for inference."""

import abc
import enum
import os
from typing import Any, List, Tuple

from absl import logging
from inference import utils
import vertexai
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import HarmCategory
from vertexai.generative_models import Part
from vertexai.generative_models import SafetySetting

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import StoppingCriteria, StoppingCriteriaList  


ContentChunk = utils.ContentChunk
MimeType = utils.MimeType
LOCATION = 'us-central1'
TEMPERATURE = 0.0


class GeminiModel(enum.StrEnum):
  GEMINI_1_5_FLASH_002 = 'gemini-1.5-flash-002'  # Max input tokens: 1,048,576
  GEMINI_1_5_PRO_002 = 'gemini-1.5-pro-002'  # Max input tokens: 2,097,152


class Model(metaclass=abc.ABCMeta):
  """Base class for models."""

  def index(
      self,
      content_chunks: List[ContentChunk],
      document_indices: List[tuple[int, int]],
      **kwargs: Any,
  ) -> str:
    """Indexes the example containing the corpus.

    Arguments:
      content_chunks: list of content chunks to send to the model.
      document_indices: list of (start, end) indices marking the documents
        boundaries within content_chunks.
      **kwargs: additional arguments to pass.

    Returns:
      Indexing result.
    """
    del content_chunks, document_indices, kwargs  # Unused.
    return 'Indexing skipped since not supported by model.'

  @abc.abstractmethod
  def infer(
      self,
      content_chunks: List[ContentChunk],
      document_indices: List[tuple[int, int]],
      **kwargs: Any,
  ) -> str:
    """Runs inference on model and returns text response.

    Arguments:
      content_chunks: list of content chunks to send to the model.
      document_indices: list of (start, end) indices marking the documents
        boundaries within content_chunks.
      **kwargs: additional arguments to pass to the model.

    Returns:
      Inference result.
    """
    raise NotImplementedError


class VertexAIModel(Model):
  """GCP VertexAI wrapper for general Gemini models."""

  def __init__(
      self,
      project_id: str,
      model_name: str,
      pid_mapper: dict[str, str],
  ):
    self.project_id = project_id
    self.model_name = model_name
    self.pid_mapper = pid_mapper
    vertexai.init(project=project_id, location=LOCATION)
    self.model = GenerativeModel(self.model_name)

  def _process_content_chunk(self, content_chunk: ContentChunk) -> Part:
    if content_chunk.mime_type in [
        MimeType.TEXT,
        MimeType.IMAGE_JPEG,
        MimeType.AUDIO_WAV,
    ]:
      return Part.from_data(
          content_chunk.data, mime_type=content_chunk.mime_type
      )
    else:
      raise ValueError(f'Unsupported MimeType: {content_chunk.mime_type}')

  def _get_safety_settings(
      self, content_chunks: List[ContentChunk]
  ) -> List[SafetySetting]:
    """Returns safety settings for the given content chunks."""
    # Audio prompts cannot use BLOCK_NONE.
    if any(
        content_chunk.mime_type == MimeType.AUDIO_WAV
        for content_chunk in content_chunks
    ):
      threshold = SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    else:
      threshold = SafetySetting.HarmBlockThreshold.BLOCK_NONE
    return [
        SafetySetting(
            category=category,
            threshold=threshold,
        )
        for category in HarmCategory
    ]

  def _postprocess_response(self, response: Any) -> List[str]:
    """Postprocesses the response from the model."""
    try:
      output_text = getattr(response, 'candidates')[0].content.parts[0].text
      final_answers = utils.extract_prediction(output_text)
      final_answers = [
          self.pid_mapper[str(answer)] for answer in final_answers
      ]
    except Exception as e:  # pylint:disable=broad-exception-caught
      logging.error('Bad response %s with error: %s', response, str(e))
      raise ValueError(f'Unexpected response: {response}') from e

    return final_answers

  def infer(
      self,
      content_chunks: List[ContentChunk],
      **kwargs: Any,
  ) -> List[str]:
    response = self.model.generate_content(
        [
            self._process_content_chunk(content_chunk)
            for content_chunk in content_chunks
        ],
        generation_config=GenerationConfig(temperature=TEMPERATURE, top_p=1.0),
        safety_settings=self._get_safety_settings(content_chunks),
    )

    return self._postprocess_response(response)


def get_model(
    model_url_or_name: str,
    project_id: str | None,
    pid_mapper: dict[str, str],
) -> Model:
  """Returns the model to use."""

  if model_url_or_name in GeminiModel.__members__.values():
    if project_id is None:
      raise ValueError(
          'Project ID and service account are required for VertexAIModel.'
      )
    model = VertexAIModel(
        project_id=project_id,
        model_name=model_url_or_name,
        pid_mapper=pid_mapper,
    )
  elif True:
      model = HuggingfaceModel(model_name=model_url_or_name,
                               pid_mapper=pid_mapper)
  else:
    raise ValueError(f'Unsupported model: {model_url_or_name}')
  return model

class HuggingfaceModel(Model):
  """Huggingface model wrapper."""
  task_prompt_list = {
    "arguana":"The following statements can counterargue the claim:",
    "fever":"The following passages can help verify this sentence:",
    "fiqa": "The following documents can answer the query:",
    "hotpotqa": "The following documents can help answer the query:",
    "msmarco": "The following documents can answer the query:",
    "musique": "The following documents are needed to answer the query:",
    "nq": "The following documents can help answer the query:",
    "qampari": "The following documents can help answer the query:",
    "quest": "The following documents are needed to answer the query:",
    "quora": "The following existing questions are most similar to the given query:",
    "scifact": "The following passages can help verify this sentence:",
    "topiocqa": "The following documents can help answer the query:",
    "webis_touche2020": "The following argument is most relevant to the query:"
  }

  def __init__(self, model_name: str, pid_mapper: dict[str, str]):
    self.model_name = model_name
    self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      torch_dtype=torch.bfloat16,
                                                      device_map="auto",
                                                      trust_remote_code=True,
                                                      attn_implementation="flash_attention_2",)

    self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                   trust_remote_code=True)
    self.pid_mapper = pid_mapper
  
  def _postprocess_response(self, response: Any) -> List[str]:
    """Postprocesses the response from the model."""
    try:
      output_text = response
      final_answers = utils.extract_prediction(output_text)
      final_answers = [
          self.pid_mapper[str(answer)] for answer in final_answers
      ]
    except Exception as e:  # pylint:disable=broad-exception-caught
      logging.error('Bad response %s with error: %s', response, str(e))
      raise ValueError(f'Unexpected response: {response}') from e

    return final_answers

  def _process_content_chunk(self, content_chunk: ContentChunk) -> str:
    if content_chunk.mime_type in [
        MimeType.TEXT,
    ]:
      return content_chunk.data.decode("utf-8")
    else:
      raise ValueError(f'Unsupported MimeType: {content_chunk.mime_type}')

  def infer(
      self,
      content_chunks: List[ContentChunk],
      **kwargs: Any,
  ) -> Tuple[List[str], str]:
    prompt = '\n'.join([self._process_content_chunk(chunk) for chunk in content_chunks])
    if os.getenv("DATASET") not in self.task_prompt_list:
      exit(10)
    prompt = prompt + "\n" + self.task_prompt_list[os.getenv("DATASET")]+"\n"
    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(self.model.device)
    output = self.model.generate(input_ids, do_sample=False, max_new_tokens=500, stopping_criteria=StoppingCriteriaList([TripleEqualsStoppingCriteria(self.tokenizer)]))
    # this is a work around to remove the prompt from the response
    # primary concern is that llama3 tokenizer does not ensures that the prompt is the same after and before tokenization
    prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    response = self.tokenizer.decode(output[0,], skip_special_tokens=True)
    if response.startswith(prompt):
      response = response[len(prompt):]
    return self._postprocess_response(response), response

  
class TripleEqualsStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequence="======"):
        self.tokenizer = tokenizer
        self.stop_sequence = stop_sequence
        self.stop_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        generated_sequence = input_ids[0].tolist()
        
        if len(generated_sequence) >= len(self.stop_ids):
            if generated_sequence[-len(self.stop_ids):] == self.stop_ids:
                return True
        
        return False