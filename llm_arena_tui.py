#!/usr/bin/env python
import os
import random
import asyncio
from typing import Dict, List, Tuple, Optional
import httpx
from urllib.parse import urlparse, urlunparse # To build the unload URL

from textual.app import App, ComposeResult, Binding
# Use specific containers for clarity
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Header, Footer, Static, Button, Input, TextArea, LoadingIndicator
)
# Ensure openai is installed: pip install "openai>=1.0"
try:
    from openai import AsyncOpenAI as OpenAIClient, APIError, APITimeoutError, RateLimitError, APIConnectionError
except ImportError:
    print("Error: 'openai' library not found or old version.")
    print("Please install/update it: pip install --upgrade \"openai>=1.0\"")
    import sys
    sys.exit(1)


# --- Configuration ---
# Replace with your OpenAI-compatible API endpoint
API_BASE_URL = os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1")
# Optional: Set API key if required by your endpoint
API_KEY = os.getenv("OPENAI_API_KEY", "NotNeeded") # Often not needed for local
# List of model identifiers EXACTLY as your API endpoint recognizes them
# --- !!! REPLACE THESE EXAMPLES !!! ---
AVAILABLE_MODELS = [
    # Check these names carefully against your running models
    # Example for Ollama via LiteLLM: "ollama/mistral"
    # Example for vLLM: "mistralai/Mistral-7B-Instruct-v0.1"
    "Gemma-3-Abliterated-Q6_K_L",         # EXAMPLE - REPLACE
    "qwq-32b-q5_k_m", # EXAMPLE - REPLACE
]
# --- End Configuration ---

class ResponseDisplay(TextArea):
    """A read-only TextArea specifically for displaying model responses."""
    DEFAULT_CSS = """
    ResponseDisplay {
        border: none;
        width: 100%;
        height: 100%;
    }
    """
    def __init__(self, title: str = "", **kwargs):
        kwargs.setdefault('show_line_numbers', False)
        super().__init__(read_only=True, name=title, **kwargs)
        self._title = title
        # We set border_title on the parent container now
        # self.border_title = self._title

# --- Main App ---
class LLMArenaApp(App[None]):
    """A Textual app for blind A/B testing of LLMs."""

    # Define CSS directly in the class
    CSS = """
    Screen {
        overflow: hidden; /* Prevent screen overflow */
    }

    /* Use Vertical layout for the main sections */
    #main-container {
        padding: 0 1; /* Padding left/right */
        height: 100%;
        width: 100%;
    }

    /* Container for the two response text areas */
    #results-area {
        height: 1fr; /* Takes up remaining vertical space */
        width: 100%;
        margin-bottom: 1; /* Space below results */
        /* border: round green; */ /* Debug */
    }

    /* Horizontal layout for the two side-by-side panels */
    #response-panels {
        height: 100%;
        width: 100%;
        /* align: center middle; */ /* Distribute space if needed */
        /* background: $panel; */ /* Debug */
    }

    /* Containers holding each TextArea for border/title */
    #response-a-container, #response-b-container {
        width: 1fr; /* Equal width */
        height: 100%;
        border: round $secondary;
        margin: 0 1; /* Space between/around panels */
        padding: 1; /* Padding inside border */
    }

    /* The actual TextAreas filling their containers */
    #response-a-text, #response-b-text {
        background: $surface; /* Match container background */
    }

    /* Container for prompt input and status bar */
    #input-status-area {
        height: auto; /* Size to content */
        width: 100%;
        margin-bottom: 1; /* Space below this area */
        /* border: round yellow; */ /* Debug */
    }

    #prompt-input {
        width: 100%;
        height: auto; /* Allow multi-line input potentially */
        max-height: 5; /* Limit input height */
        margin-bottom: 1;
    }

    #status-bar {
        width: 100%;
        height: 1; /* Fixed height */
        text-style: dim;
        overflow: hidden;
    }

    /* Container for action buttons */
    #action-buttons {
        height: auto;
        width: 100%;
        align-horizontal: center; /* Center buttons */
        /* border: round blue; */ /* Debug */
    }

    #action-buttons Button {
        margin: 0 1;
        width: auto;
        min-width: 10;
    }

    /* Loading Indicator */
    #loading-indicator-container {
        align: center middle;
        height: auto;
        width: 100%;
        margin-top: 1;
        /* background: $panel-darken-2; */ /* Optional semi-transparent background */
    }

    LoadingIndicator {
        /* Style the indicator itself if needed */
    }
    """

    BINDINGS = [
        Binding("ctrl+d", "quit", "Quit"),
        Binding("ctrl+r", "reshuffle", "New Round / Reshuffle", show=True),
#        Binding("enter", "submit_prompt", "Submit", show=False), # Use Enter specifically
    ]

    # --- Reactive State Variables ---
    # (Keep previous reactive variables)
    current_prompt = reactive("")
    response_a = reactive("")
    response_b = reactive("")
    model_a_name = reactive("?")
    model_b_name = reactive("?")
    app_state = reactive("idle") # "idle", "loading", "showing_results", "voted", "error"
    status_message = reactive("Enter a prompt to start.")
    round_count = reactive(0)
    preferences: Dict[str, Dict[str, int]] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._selected_models: Optional[Tuple[str, str]] = None
        self._client: Optional[OpenAIClient] = None

    @staticmethod
    def _initialize_openai_client() -> Optional[OpenAIClient]:
        """Initializes the OpenAI client."""
        try:
             client = OpenAIClient(base_url=API_BASE_URL, api_key=API_KEY, timeout=120.0)
             return client
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}") # Log this better in real app
            return None

    # --- Compose Method (Refactored Layout) ---
    def compose(self) -> ComposeResult:
        yield Header()
        # Main vertical layout container
        with Vertical(id="main-container"):
            # Area for results panels (takes remaining space)
            with Container(id="results-area"):
                with Horizontal(id="response-panels"):
                    # Container A for border/title
                    with Container(id="response-a-container"):
                        yield ResponseDisplay("Model A", id="response-a-text")
                    # Container B for border/title
                    with Container(id="response-b-container"):
                        yield ResponseDisplay("Model B", id="response-b-text")
            # Area for input and status
            with Vertical(id="input-status-area"):
                 yield Input(placeholder="Enter your prompt here...", id="prompt-input")
                 yield Static(id="status-bar", expand=True)
            # Area for buttons
            with Horizontal(id="action-buttons"):
                 yield Button("Vote A", id="vote-a", variant="primary", disabled=True)
                 yield Button("Vote B", id="vote-b", variant="primary", disabled=True)
                 yield Button("Tie", id="vote-tie", disabled=True)
                 yield Button("Both Bad", id="vote-bad", disabled=True)
                 yield Button("Continue", id="continue", variant="success", disabled=True)
            # Separate container for loading indicator, initially hidden
            with Container(id="loading-indicator-container", classes="hidden"):
                 yield LoadingIndicator()
        yield Footer()

    # --- Event Handlers and Logic (Mostly Unchanged, adapt watchers) ---

    def on_mount(self) -> None:
        """Called when the app is first mounted."""
        if not AVAILABLE_MODELS or len(AVAILABLE_MODELS) < 2:
            self.notify("Error: At least two models must be defined in AVAILABLE_MODELS.", title="Config Error", severity="error")
            self.exit(); return

        self._client = self._initialize_openai_client()
        if self._client is None:
            self.notify(f"Failed to initialize OpenAI client. Check URL: {API_BASE_URL}", title="API Error", severity="error")
            self.exit(); return

        # Initially hide the loading indicator container
        self.set_loading(False)
        self.shuffle_models()

    def set_loading(self, is_loading: bool) -> None:
        """Show/hide the loading indicator."""
        loading_container = self.query_one("#loading-indicator-container")
        if is_loading:
            loading_container.remove_class("hidden")
            # Potentially hide other elements if needed, e.g., input/buttons
            # self.query_one("#input-status-area").add_class("hidden")
            # self.query_one("#action-buttons").add_class("hidden")
            self.query_one("#prompt-input").disabled = True
            self.set_voting_buttons_disabled(True)
            self.query_one("#continue").disabled = True
        else:
            loading_container.add_class("hidden")
            # Restore visibility if hidden
            # self.query_one("#input-status-area").remove_class("hidden")
            # self.query_one("#action-buttons").remove_class("hidden")
            # Enable input only when appropriate (idle, voted)
            self.query_one("#prompt-input").disabled = self.app_state not in ("idle", "voted")


    def shuffle_models(self) -> None:
        """Randomly selects two different models and assigns them to A and B."""
        if len(AVAILABLE_MODELS) < 2:
             self.status_message = "Need at least 2 models to test."
             return
        chosen = random.sample(AVAILABLE_MODELS, 2)
        self._selected_models = (chosen[0], chosen[1]) if random.random() > 0.5 else (chosen[1], chosen[0])
        self.model_a_name = "?"
        self.model_b_name = "?"
        self.round_count += 1
        self.reset_ui_for_new_prompt()
        self.status_message = f"Round {self.round_count}. Enter prompt. [ Models Shuffled ]"

    # --- State Update Helpers (Adapt loading state) ---

    def set_state_loading(self) -> None:
        self.app_state = "loading"
        self.set_loading(True)
        self.status_message = "Generating responses..."
        self.response_a = ""
        self.response_b = ""

    def set_state_showing_results(self) -> None:
        self.app_state = "showing_results"
        self.set_loading(False) # Hide loader
        self.set_voting_buttons_disabled(False) # Enable voting
        self.query_one("#continue").disabled = True # Keep continue disabled
        self.query_one("#prompt-input").disabled = True # Keep prompt disabled
        self.status_message = "Responses received. Please vote or press Ctrl+R to re-roll."
        try: self.query_one("#vote-a").focus()
        except Exception: pass

    def set_state_voted(self, winner: str) -> None:
        self.app_state = "voted"
        self.set_loading(False) # Ensure loader is hidden
        if self._selected_models:
            self.model_a_name = self._selected_models[0]
            self.model_b_name = self._selected_models[1]
            self.update_preference_tally(winner)
            self.status_message = self.get_tally_string() + " | Press 'Continue' or Ctrl+R."
        else:
             self.status_message = "Error: Models not selected. Press Ctrl+R."
             self.log.error("set_state_voted called but _selected_models is None")
        self.set_voting_buttons_disabled(True)
        self.query_one("#continue").disabled = False
        self.query_one("#prompt-input").disabled = False # Allow prompt input now
        self.query_one("#continue").focus()

    def set_state_error(self, error_message: str) -> None:
        self.app_state = "error"
        self.set_loading(False)
        self.set_voting_buttons_disabled(True)
        self.query_one("#continue").disabled = True
        self.query_one("#prompt-input").disabled = True # Keep disabled on error
        self.status_message = f"Error: {error_message}. Press Ctrl+R for new round, Ctrl+D to quit."
        self.notify(f"Error: {error_message}", title="API Error", severity="error", timeout=6)

    def reset_ui_for_new_prompt(self) -> None:
        self.app_state = "idle"
        self.set_loading(False) # Ensure loader hidden
        self.query_one("#response-a-text", TextArea).clear()
        self.query_one("#response-b-text", TextArea).clear()
        self.response_a = "" # Reset reactive vars
        self.response_b = ""
        self.query_one("#prompt-input").disabled = False
        self.query_one("#prompt-input").clear()
        self.query_one("#prompt-input").focus()
        self.set_voting_buttons_disabled(True)
        self.query_one("#continue").disabled = True
        if self.model_a_name != "?":
             self.model_a_name = "?"
             self.model_b_name = "?"
        self.status_message = f"Round {self.round_count}. Enter prompt."

    def set_voting_buttons_disabled(self, disabled: bool) -> None:
        self.query_one("#vote-a").disabled = disabled
        self.query_one("#vote-b").disabled = disabled
        self.query_one("#vote-tie").disabled = disabled
        self.query_one("#vote-bad").disabled = disabled

    # --- Watchers (Adapt selectors) ---

    def watch_response_a(self, new_response: str) -> None:
        self.log.debug(f"Watcher watch_response_a triggered. Length: {len(new_response)}") # ADDED LOG        
        try:
            widget_a = self.query_one("#response-a-text", TextArea)
            widget_a.load_text(new_response)
            self.log.debug("watch_response_a: load_text called.") # ADDED LOG
        except Exception as e:
            self.log.exception("Error updating response A text area!") # ADDED LOG

    def watch_response_b(self, new_response: str) -> None:
        self.log.debug(f"Watcher watch_response_b triggered. Length: {len(new_response)}") # ADDED LOG
        try:
            widget_b = self.query_one("#response-b-text", TextArea)
            widget_b.load_text(new_response)
            self.log.debug("watch_response_b: load_text called.") # ADDED LOG
        except Exception as e:
            self.log.exception("Error updating response B text area!") # ADDED LOG

    def watch_model_a_name(self, new_name: str) -> None:
        # Target the container holding the text area for the border title
        self.query_one("#response-a-container", Container).border_title = f"Model A ({new_name})"

    def watch_model_b_name(self, new_name: str) -> None:
        self.query_one("#response-b-container", Container).border_title = f"Model B ({new_name})"

    def watch_status_message(self, new_message: str) -> None:
        self.query_one("#status-bar", Static).update(new_message)

    # --- Actions and Event Handlers (Adapt prompt submission) ---

    # Bind Enter explicitly
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Called when Enter is pressed in the input."""
        self.log.debug(f"Input submitted. Current state: {self.app_state}. Value: '{event.value}'") # Updated log
        if self.app_state in ("idle", "voted"):
            self.action_send_prompt()
        elif self.app_state == "showing_results":
            # Maybe focus voting buttons instead?
            try: self.query_one("#vote-a").focus()
            except Exception: pass
        else:
            self.log.warning(f"Input submission ignored due to app state: {self.app_state}")
    # Keep action_send_prompt for potential Ctrl+S binding or direct calls
    def action_send_prompt(self) -> None:
        """Initiates sending the prompt."""
        prompt = self.query_one("#prompt-input", Input).value.strip()
        if not prompt:
            self.notify("Prompt cannot be empty.", severity="warning", timeout=3)
            return
        if self.app_state == "loading":
            self.notify("Already generating response.", severity="warning", timeout=3)
            return
        if not self._client:
             self.set_state_error("API Client not initialized"); return
        if not self._selected_models:
             self.set_state_error("Internal Error: Models not selected"); return

        self.current_prompt = prompt
        self.run_worker(self.generate_responses(prompt), exclusive=True)

    async def generate_responses(self, prompt: str) -> None:
        """The async worker task to fetch responses."""
        self.set_state_loading()
        # Ensure models are selected (should be checked before calling worker)
        if not self._selected_models:
            self.log.critical("generate_responses called with _selected_models=None")
            self.set_state_error("Internal error (models not selected)")
            return
        model_a, model_b = self._selected_models

        results = await asyncio.gather(
            self.fetch_llm_response(model_a, prompt),
            self.fetch_llm_response(model_b, prompt),
            return_exceptions=True
        )
        res_a, res_b = results

        critical_error = False
        error_details = []

        if isinstance(res_a, Exception):
             err_msg = f"A: {type(res_a).__name__}"
             self.response_a = f"Error (Model A):\n{type(res_a).__name__}: {res_a}"
             if isinstance(res_a, (APIConnectionError, APITimeoutError)):
                 critical_error = True; error_details.append(err_msg)
             self.log.error(f"API Error (A: {model_a}): {res_a}")
        else: self.response_a = res_a

        if isinstance(res_b, Exception):
             err_msg = f"B: {type(res_b).__name__}"
             self.response_b = f"Error (Model B):\n{type(res_b).__name__}: {res_b}"
             if isinstance(res_b, (APIConnectionError, APITimeoutError)):
                 critical_error = True; error_details.append(err_msg)
             self.log.error(f"API Error (B: {model_b}): {res_b}")
        else: self.response_b = res_b

        if critical_error:
             self.set_state_error(f"API Connection/Timeout ({', '.join(error_details)})")
        else:
             self.set_state_showing_results()

    async def fetch_llm_response(self, model_name: str, prompt: str) -> str:
        """Calls the OpenAI-compatible API (identical logic to previous)"""
        assert self._client is not None
        self.log.info(f"Sending prompt to model: {model_name}")
        try:
            response = await self._client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=2048, stream=False,
            )
            self.log.info(f"Received response from model: {model_name}: {response}")
            if response.choices and response.choices[0].message:
                 content = response.choices[0].message.content
                 return content.strip() if content else "[No content]"
            return "[Invalid response structure]"
        except (APIConnectionError, APITimeoutError) as e:
             self.log.error(f"API Conn/Timeout Error ({model_name}): {e}")
             raise
        except RateLimitError as e:
             self.log.warning(f"API Rate Limit Error ({model_name}): {e}")
             raise Exception(f"Rate Limit: {e}") from e
        except APIError as e:
             self.log.error(f"API Error ({model_name}, {e.status_code}): {getattr(e, 'body', None) or e}")
             err_body = getattr(e, 'body', {}) or {}
             err_msg = err_body.get('message', str(e)) if isinstance(err_body, dict) else str(e)
             # Check for common 'model not found' patterns
             if "not found" in err_msg.lower() or "exist" in err_msg.lower():
                  raise Exception(f"Model not found: {model_name}") from e
             raise Exception(f"API Error ({e.status_code}): {err_msg}") from e
        except Exception as e:
            self.log.exception(f"Unexpected Error fetching ({model_name})")
            raise Exception(f"Client Error: {e}") from e

    async def unload_model(self, model_name: str) -> None:
        """Sends a request to the server to unload a model."""
        if not model_name:
            self.log.warning("Unload skipped: No model name provided.")
            return

        try:
            # Construct the unload URL (e.g., http://host:port/unload)
            base_url_parts = urlparse(API_BASE_URL)
            # Assume unload is at the root path, replace /v1 or other paths
            unload_url = urlunparse((base_url_parts.scheme, base_url_parts.netloc, "/unload", '', '', ''))
            # Alternative if it's relative to /v1:
            # unload_url = API_BASE_URL.rstrip('/') + '/unload' # Check llama-swap docs

            payload = {"model": model_name}
            self.log.info(f"--> Attempting to unload model '{model_name}' via POST to {unload_url}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(unload_url, json=payload)

            response.raise_for_status() # Raise exception for 4xx/5xx errors
            self.log.info(f"<-- Successfully unloaded model '{model_name}' (Status: {response.status_code})")

        except httpx.RequestError as e:
            self.log.error(f"!!! Unload Error (Network/Request): Failed to unload '{model_name}'. URL: {e.request.url}. Error: {e}")
            # Notify user? Might be too noisy.
        except httpx.HTTPStatusError as e:
            self.log.error(f"!!! Unload Error (HTTP Status): Failed to unload '{model_name}'. Status: {e.response.status_code}. Response: {e.response.text}")
        except Exception as e:
            self.log.exception(f"!!! Unload Error (Unexpected): Failed to unload '{model_name}'.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks (identical logic to previous)"""
        button_id = event.button.id
        if self.app_state == "showing_results":
            if button_id == "vote-a": self.set_state_voted(winner="A")
            elif button_id == "vote-b": self.set_state_voted(winner="B")
            elif button_id == "vote-tie": self.set_state_voted(winner="tie")
            elif button_id == "vote-bad": self.set_state_voted(winner="both_bad")
        elif self.app_state == "voted" and button_id == "continue":
             self.action_reshuffle()

    def action_reshuffle(self) -> None:
        """Starts a new round (identical logic to previous)"""
        if self.app_state in ("voted", "showing_results", "error"):
             self.log.info("Reshuffling models for new round.")
             self.shuffle_models()
        elif self.app_state == "loading":
             self.notify("Cannot start new round while loading.", severity="warning", timeout=3)

    # --- Preference Tally Logic (Identical to previous) ---
    def get_preference_key(self) -> Optional[str]:
        if not self._selected_models: return None
        model1, model2 = sorted(self._selected_models)
        return f"{model1}_vs_{model2}"

    def update_preference_tally(self, winner: str) -> None:
        if not self._selected_models:
             self.log.error("Tally update skipped: No models selected.")
             return
        key = self.get_preference_key()
        if not key: return

        scores = self.preferences.setdefault(key, {"model1_wins": 0, "model2_wins": 0, "ties": 0, "both_bad": 0})
        model1_sorted, _ = sorted(self._selected_models)
        actual_model_a, actual_model_b = self._selected_models

        if winner == "A": scores["model1_wins" if actual_model_a == model1_sorted else "model2_wins"] += 1
        elif winner == "B": scores["model1_wins" if actual_model_b == model1_sorted else "model2_wins"] += 1
        elif winner == "tie": scores["ties"] += 1
        elif winner == "both_bad": scores["both_bad"] += 1
        self.log.info(f"Vote recorded for {key}: Winner={winner}. New scores: {scores}")

    def get_tally_string(self) -> str:
        if not self._selected_models or self.app_state != "voted":
            return self.status_message # Return current status if not showing tally
        key = self.get_preference_key()
        if not key or key not in self.preferences:
            return "[No votes recorded for this pair yet]"

        scores = self.preferences[key]
        model1_sorted, model2_sorted = sorted(self._selected_models)
        m1_display = model1_sorted.split('/')[-1].split(':')[-1]
        m2_display = model2_sorted.split('/')[-1].split(':')[-1]
        max_len = 25
        if len(m1_display) > max_len: m1_display = m1_display[:max_len-1] + "…"
        if len(m2_display) > max_len: m2_display = m2_display[:max_len-1] + "…"
        return (f"Tally ({m1_display} vs {m2_display}): "
                f"M1 Wins: {scores['model1_wins']} | M2 Wins: {scores['model2_wins']} | "
                f"Ties: {scores['ties']} | Bad: {scores['both_bad']}")


if __name__ == "__main__":
    # CSS is now defined inside the App class
    # No need to write file
    app = LLMArenaApp()
    app.run()
