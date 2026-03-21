# Grok API Chat Completions Request Body

This README documents the structure of the request body for the Grok API chat completions endpoint, based on the provided schema. The schema defines the parameters used to interact with the model, including inputs, configurations, and tools. Fields are described with their types, requirements, defaults (where applicable), and nested structures.

The API supports various input types (e.g., text, images, files) and advanced features like tool calls, reasoning configurations, and search parameters. Note that some fields are included for compatibility reasons and may not be fully supported.

## Top-Level Fields

### input
- **Type**: string | array
- **Required**: Yes
- **Description**: The input passed to the model. Can be text, image, or file.

  One of:
  - **string**: Text input.
  - **array**: A list of input items to the model. Can be of different types.

    One of:
    - **object**: Message input to the model.
      - **content** (string | array, required): Text, image, or audio input.
        One of:
        - **string**: Text input.
        - **array**: A list of input items to the model. Can include text and images.
          One of:
          - **object**: Text input.
            - **text** (string, required): Text input.
            - **type** ("input_text", required).
          - **object**: Image input. Note: Storing and fetching images is not fully supported at the moment.
            - **detail** (string | null): Specifies the detail level of the image. One of "high", "low", or "auto". Defaults to "auto".
            - **file_id** (string | null): Only included for compatibility.
            - **image_url** (string, required): A public URL of image prompt, only available for vision models.
            - **type** ("input_image", required).
          - **object**: File input.
            - **file_id** (string, required): The file ID from the Files API.
            - **type** ("input_file", required).
      - **name** (string | null): A unique identifier representing your end-user, which can help xAI to monitor and detect abuse. Only supported for "user" messages.
      - **role** (string, required): The role of the message. Possible values are "user", "assistant", "system", and "developer".
      - **type** (string | null): The type of the message, which is always "message".
    - Previous responses of the model and tool call outputs.
      All of:
      - Required.
      One of:
      - All of:
        - Required.
        One of:
        - **object**:
          All of:
          - **object** (required): An output message from the model.
            - **content** (array, required): Content of the output message.
              One of:
              - **object**: Text output.
                - **annotations** (array, required): Citations.
                  - **end_index** (integer | null): The end index of the annotation.
                  - **start_index** (integer | null): The start index of the annotation.
                  - **title** (string | null): The title of the annotation.
                  - **type** (string, required): The type of the annotation. Only supported type currently is "url_citation".
                  - **url** (string, required): The URL of the web resource.
                - **logprobs** (array, required): The log probabilities of each output token returned in the content of message.
                  - **bytes** (array | null): The ASCII encoding of the output character.
                    - integer (min: 0, required).
                  - **logprob** (number, required): The log probability of returning this token.
                  - **token** (string, required): The token.
                  - **top_logprobs** (array, required): An array of the most likely tokens to return at this token position.
                    - **bytes** (array | null): The ASCII encoding of the output character.
                      - integer (min: 0, required).
                    - **logprob** (number, required): The log probability of returning this token.
                    - **token** (string, required): The token.
                - **text** (string, required): The text output from the model.
                - **type** ("output_text", required).
              - **object**: Refusal.
                - **refusal** (string, required): The reason for the refusal.
                - **type** ("refusal", required).
            - **id** (string, required): The unique ID of the output message.
            - **role** (string, required): The role of the output message, which can be "assistant" or "tool".
            - **status** (string, required): Status of the item. One of "completed", "in_progress", or "incomplete".
            - **type** (string, required): The type of the output message, which is always "message".
          - **object**:
            All of:
            - **object** (required): A tool call to run a function.
              - **arguments** (string, required): The arguments to pass to the function, as a JSON string.
              - **call_id** (string, required): The unique ID of the function tool call generated by the model.
              - **id** (string, required): The unique ID of the function tool call.
              - **name** (string, required): The name of the function.
              - **status** (string, required): Status of the item. One of "completed", "in_progress", or "incomplete".
              - **type** (string, required): The type of the function tool call, which can be "function_call" for client-side tool calls, and "web_search_call" or "x_search_call" or "code_interpreter_call" or "mcp_call" for server-side tool calls.
          - **object**:
            All of:
            - **object** (required): The reasoning done by the model.
              - **encrypted_content** (string | null): The encrypted reasoning. Returned when "reasoning.encrypted_content" is passed in "include".
              - **id** (string, required): The unique ID of the reasoning content.
              - **status** (string, required): Status of the item. One of "completed", "in_progress", or "incomplete".
              - **summary** (array, required): The reasoning text contents.
                - **text** (string, required): Reasoning done by the model.
                - **type** (string, required): The type of the object, which is always "summary_text".
              - **type** (string, required): The type of the object, which is always "reasoning".
          - **object**:
            All of:
            - **object** (required): The output of a web search tool call.
              - **action** (object, required): An object describing the specific action taken in this web search call. Includes details on how the model used the web (search, open_page, find).
                One of:
                - **object**: Action type "search" - Performs a web search query.
                  - **query** (string, required): The search query.
                  - **sources** (array, required): The sources used in the search.
                    - **type** (string, required): The type of source.
                    - **url** (string, required): The URL of the source.
                  - **type** ("search", required).
                - **object**: Action type "open_page" - Opens a specific URL from search results.
                  - **type** ("open_page", required).
                  - **url** (string, required): The URL of the page to open.
                - **object**: Action type "find": Searches for a pattern within a loaded page.
                  - **pattern** (string, required): The pattern or text to search for within the page.
                  - **source** (object, required): The source of the page to search in.
                    - **type** (string, required): The type of source.
                    - **url** (string, required): The URL of the source.
                  - **type** ("find", required).
              - **id** (string, required): The unique ID of the web search tool call.
              - **status** (string, required): The status of the web search tool call.
              - **type** (string, required): The type of the web search tool call. Always "web_search_call".
          - **object**:
            All of:
            - **object** (required): The output of a web search tool call.
              - **id** (string, required): The unique ID of the file search tool call.
              - **queries** (array, required): The queries used to search for files.
                - string (required).
              - **results** (array, required): The results of the file search tool call.
                - **file_id** (string, required): The file ID of the file search result.
                - **filename** (string, required): The filename of the file search result.
                - **score** (number, required): The score of the file search result.
                - **text** (string, required): The text of the file search result.
              - **status** (string, required): The status of the file search tool call.
              - **type** (string, required): The type of the file search tool call. Always "file_search_call".
          - **object**:
            All of:
            - **object** (required): The output of a code interpreter tool call.
              - **code** (string, required): The code of the code interpreter tool call.
              - **id** (string, required): The unique ID of the code interpreter tool call.
              - **outputs** (array, required): The outputs of the code interpreter tool call.
                One of:
                - **object**: The output of the code interpreter tool call.
                  - **logs** (string, required): The output of the code interpreter tool call.
                  - **type** ("logs", required).
                - **object**: The error of the code interpreter tool call.
                  - **type** ("image", required).
                  - **url** (string, required): The error of the code interpreter tool call.
              - **status** (string, required): The status of the code interpreter tool call.
              - **type** (string, required): The type of the code interpreter tool call. Always "code_interpreter_call".
          - **object**:
            All of:
            - **object** (required): The output of a MCP tool call.
              - **arguments** (string, required): A JSON string of the arguments passed to the tool.
              - **error** (string, required): The error message of the MCP tool call.
              - **id** (string, required): The unique ID of the MCP tool call.
              - **name** (string, required): The name of the tool that was run.
              - **output** (string, required): The output of the MCP tool call.
              - **server_label** (string, required): The label of the MCP server running the tool.
              - **status** (string, required): The status of the MCP tool call.
              - **type** (string, required): The type of the MCP tool call. Always "mcp_call".
          - **object**:
            All of:
            - **object** (required): The output of a custom tool call.
              - **call_id** (string, required): The unique ID of the function tool call generated by the model.
              - **id** (string, required): The status of the custom tool call.
              - **input** (string, required): The unique ID of the custom tool call.
              - **name** (string, required): An identifier used to map this custom tool call to a tool call output.
              - **status** (string, required): Status of the item. One of "completed", "in_progress", or "incomplete".
              - **type** (string, required): The input for the custom tool call generated by the model.
          - **object**:
            All of:
            - **object** (required): The output of a function tool call.
              - **call_id** (string, required): The unique ID of the function tool call generated by the model.
              - **output** (string, required): The output of the function tool call, as a JSON string.
              - **type** (string, required): The type of the function tool call, which is always "function_call_output".

### background
- **Type**: boolean | null
- **Default**: false
- **Description**: (Unsupported) Whether to process the response asynchronously in the background.

### include
- **Type**: array | null
- **Description**: What additional output data to include in the response. Currently the only supported value is "reasoning.encrypted_content" which returns an encrypted version of the reasoning tokens.
  - string (required).

### instructions
- **Type**: string | null
- **Description**: An alternate way to specify the system prompt. Note that this cannot be used alongside "previous_response_id", where the system prompt of the previous message will be used.

### logprobs
- **Type**: boolean | null
- **Default**: false
- **Description**: Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.

### max_output_tokens
- **Type**: integer | null
- **Description**: Max number of tokens that can be generated in a response. This includes both output and reasoning tokens.

### metadata
- **Description**: Not supported. Only maintained for compatibility reasons.

### model
- **Type**: string
- **Description**: Model name for the model to use. Obtainable from <https://console.x.ai/team/default/models> or <https://docs.x.ai/docs/models>.

### parallel_tool_calls
- **Type**: boolean | null
- **Default**: true
- **Description**: Whether to allow the model to run parallel tool calls.

### previous_response_id
- **Type**: string | null
- **Description**: The ID of the previous response from the model.

### reasoning
- **Type**: null | object
- **Description**: Reasoning configuration. Only for reasoning models.
  - **object**:
    - **effort** (string | null, default: medium): Constrains how hard a reasoning model thinks before responding. Possible values are "low" (uses fewer reasoning tokens), "medium", and "high" (uses more reasoning tokens).
    - **generate_summary** (string | null): Only included for compatibility.
    - **summary** (string | null): A summary of the model's reasoning process. Possible values are "auto", "concise", and "detailed". Only included for compatibility. The model shall always return "detailed".

### search_parameters
- **Type**: null | object
- **Description**: Set the parameters to be used for searched data. Takes precedence over "web_search_preview" tool if specified in the tools.
  - **object**:
    - **from_date** (string | null): Date from which to consider the results in ISO-8601 YYYY-MM-DD. See <https://grok.x.ai/grokipedia/iso_8601>.
    - **max_search_results** (integer | null, default: 15, min: 1, max: 30): Maximum number of search results to use.
    - **mode** (string | null, default: auto): Choose the mode to query realtime data: "off": no search performed and no external will be considered. "on" (default): the model will search in every sources for relevant data. "auto": the model choose whether to search data or not and where to search the data.
    - **return_citations** (boolean | null, default: true): Whether to return citations in the response or not.
    - **sources** (array | null): List of sources to search in. If no sources specified, the model will look over the web and X by default.
      - **object** (required):
        - **excluded_x_handles** (array | null): List of X handles to exclude from the search results. X posts returned will not include any posts authored by these handles.
          - string (required).
        - **included_x_handles** (array | null): NOTE: "included_x_handles" and "x_handles" are the same parameter. "included_x_handles" is the new name but we keep both for backward compatibility. X Handles of the users from whom to consider the posts. Only available if mode is "auto", "on", or "x".
          - string (required).
        - **post_favorite_count** (integer | null): The minimum favorite count of the X posts to consider.
        - **post_view_count** (integer | null): The minimum view count of the X posts to consider.
        - **type** ("x", required).
        - **x_handles** (array | null): DEPRECATED in favor of "included_x_handles". Use "included_x_handles" instead. X Handles of the users from whom to consider the posts. Only available if mode is "auto", "on", or "x".
          - string (required).
      - **object** (required):
        - **allowed_websites** (array | null): List of website to allow in the search results. This parameter act as a whitelist where only those websites can be selected. A maximum of 5 websites can be selected. Note 1: If no relevant information is found on those websites, the number of results returned might be smaller than "max_search_results". Note 2: This parameter cannot be set with "excluded_websites".
          - string (required).
        - **country** (string | null): ISO alpha-2 code of the country. If the country is set, only data coming from this country will be considered. See <https://grok.x.ai/grokipedia/iso_3166-2>.
        - **excluded_websites** (array | null): List of website to exclude from the search results without protocol specification or subdomains. A maximum of 5 websites can be excluded. Note 2: This parameter cannot be set with "allowed_websites".
          - string (required).
        - **safe_search** (boolean | null): If set to true, mature content won't be considered during the search. Default to true.
        - **type** ("web", required).
      - **object** (required):
        - **country** (string | null): ISO alpha-2 code of the country. If the country is set, only data coming from this country will be considered. See <https://grok.x.ai/grokipedia/iso_3166-2>.
        - **excluded_websites** (array | null): List of website to exclude from the search results without protocol specification or subdomains. A maximum of 5 websites can be excluded.
          - string (required).
        - **safe_search** (boolean | null): If set to true, mature content won't be considered during the search. Default to true.
        - **type** ("news", required).
      - **object** (required):
        - **links** (array, required): Links of the RSS feeds.
          - string (required).
        - **type** ("rss", required).
    - **to_date** (string | null): Date up to which to consider the results in ISO-8601 YYYY-MM-DD. See <https://grok.x.ai/grokipedia/iso_8601>.

### service_tier
- **Type**: string | null
- **Description**: Not supported. Only maintained for compatibility reasons.

### store
- **Type**: boolean | null
- **Default**: true
- **Description**: Whether to store the input message(s) and model response for later retrieval.

### stream
- **Type**: boolean | null
- **Default**: false
- **Description**: If set, partial message deltas will be sent. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a "data: [DONE]" message.

### temperature
- **Type**: number | null
- **Default**: 1
- **Min**: 0
- **Max**: 2
- **Description**: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.

### text
- **Type**: null | object
- **Description**: Settings for customising a text response from the model.
  - **object**:
    - **format** (null | object): An object specifying the format that the model must output. Specify { "type": "json_object" } for JSON output, or { "type": "json_schema", "json_schema": {...} } for structured outputs. If { "type": "text" }, the model will return a text response.
      One of:
      - **object**: Specify text response format, always "text".
        - **type** ("text", required).
      - **object**: Specify json_object response format, always "json_object". Used for backward compatibility. Prefer to use "json_schema" instead of this.
        - **type** ("json_object", required).
      - **object**: Specify json_schema response format with a given schema. Type is always "json_schema".
        - **description** (string | null): Only included for compatibility.
        - **name** (string | null): Only included for compatibility.
        - **schema** (required): A json schema representing the desired response schema.
        - **strict** (boolean | null): Only included for compatibility.
        - **type** ("json_schema", required).

### tool_choice
- **Type**: null | string | object
- **Description**: Controls which (if any) tool is called by the model. "none" means the model will not call any tool and instead generates a message. "auto" means the model can pick between generating a message or calling one or more tools. "required" means the model must call one or more tools. Specifying a particular tool via {"type": "function", "function": {"name": "my_function"}} forces the model to call that tool. "none" is the default when no tools are present. "auto" is the default if tools are present.
  One of:
  - **string**: Controls tool access by the model. "none" makes model ignore tools, "auto" let the model automatically decide whether to call a tool, "required" forces model to pick a tool to call.
  - **object**:
    - **name** (string, required): Name of the function to use.
    - **type** (string, required): Type is always "function".

### tools
- **Type**: array | null
- **Description**: A list of tools the model may call in JSON-schema. Currently, only functions and web search are supported as tools. A max of 128 tools are supported. "web_search_preview" tool, if specified, will be overridden by "search_parameters".
  - **object** (required): A function that the model can call.
    All of:
    - **object** (required): Definition of the tool call made available to the model.
      - **description** (string | null): A description of the function to indicate to the model when to call it.
      - **name** (string, required): The name of the function. If the model calls the function, this name is used in the response.
      - **parameters** (required): A JSON schema describing the function parameters. The model _should_ follow the schema, however, this is not enforced at the moment.
      - **strict** (boolean | null): Not supported. Only maintained for compatibility reasons.
    - **object** (required): A function that the model can call.
      - **type** ("function", required).
  - **object** (required): Search the web.
    All of:
    - **object** (required):
      - **filters** (required): Only included for compatibility.
      - **search_context_size** (string | null, default: medium): This field included for compatibility reason with OpenAI's API. It is mapped to "max_search".
      - **user_location** (required): Only included for compatibility.
    - **object** (required):
      - **allowed_domains** (array | null): List of website domains to allow in the search results. This parameter act as a whitelist where only those websites can be selected. A maximum of 5 websites can be selected. Note: This parameter cannot be set with "excluded_domains".
        - string (required).
      - **enable_image_understanding** (boolean | null): Enable image understanding during web search.
      - **excluded_domains** (array | null): List of website domains to exclude from the search results without protocol specification or subdomains. A maximum of 5 websites can be excluded. Note: This parameter cannot be set with "allowed_domains".
        - string (required).
      - **external_web_access** (boolean | null): Control whether the web search tool fetches live content or uses only cached content. For OpenAI API compatibility ONLY. Request will be rejected if this field is set.
      - **filters** (null | object): Filters to apply to the search results. Compatible with OpenAI's API.
        - **object**:
          - **allowed_domains** (array | null): List of website domains (without protocol specification or subdomains) to restrict search results to (e.g., ["example.com"]). A maximum of 5 websites can be allowed. Use this as a whitelist to limit results to only these specific sites; no other websites will be considered. If no relevant information is found on these websites, the number of results returned might be smaller than "max_search_results" set in "SearchParameters". Note: This parameter cannot be set together with "excluded_domains".
            - string (required).
          - **excluded_domains** (array | null): List of website domains (without protocol specification or subdomains) to exclude from search results (e.g., ["example.com"]). Use this to prevent results from unwanted sites. A maximum of 5 websites can be excluded. This parameter cannot be set together with "allowed_domains".
            - string (required).
          - **search_context_size** (string | null): High level guidance for the amount of context window space to use for the search. Available values are "low", "medium", or "high". For OpenAI API compatibility ONLY. Request will be rejected if this field is set.
          - **user_location** (null | object): The user location to use for the search. For OpenAI API compatibility ONLY. Request will be rejected if this field is set.
            - **object**:
              - **city** (string | null): City of the user's location.
              - **country** (string | null): Two-letter ISO 3166-1 alpha-2 country code, like US, GB, etc.
              - **region** (string | null): Region of the user's location.
              - **timezone** (string | null): Timezone of the user's location, IANA timezone like America/Chicago, Europe/London, etc.
              - **type** (string, required): Type is always "approximate".
    - **object** (required): Search the web.
      - **type** ("web_search", required).
  - **object** (required): Search X.
    - **allowed_x_handles** (array | null): List of X Handles of the users from whom to consider the posts. Note: This parameter cannot be set with "excluded_x_handles".
      - string (required).
    - **enable_image_understanding** (boolean | null): Enable image understanding during X search.
    - **enable_video_understanding** (boolean | null): Enable video understanding during X search.
    - **excluded_x_handles** (array | null): List of X Handles of the users from whom to exclude the posts. Note: This parameter cannot be set with "allowed_x_handles".
      - string (required).
    - **from_date** (string | null): Date from which to consider the results in ISO-8601 YYYY-MM-DD. See <https://grok.x.ai/grokipedia/iso_8601>.
    - **to_date** (string | null): Date up to which to consider the results in ISO-8601 YYYY-MM-DD. See <https://grok.x.ai/grokipedia/iso_8601>.
    - **type** ("x_search", required).
  - **object** (required): Search the knowledge bases.
    - **filters** (required): A filter to apply. For OpenAI API compatibility ONLY. Request will be rejected if this field is set.
    - **max_num_results** (integer | null, min: 1).
    - **ranking_options** (required): Ranking options for search. For OpenAI API compatibility ONLY. Request will be rejected if this field is set.
    - **type** ("file_search", required).
    - **vector_store_ids** (array, required): List of vector store IDs to search within.
      - string (required).
  - **object** (required): Execute code.
    - **container** (required): The code interpreter container. Can be a container ID or an object that specifies uploaded file IDs to make available to your code. For OpenAI API compatibility ONLY. Request will be rejected if this field is set.
    - **type** ("code_interpreter", required).
  - **object** (required): A remote MCP server to use.
    - **allowed_tools** (array | null).
      - string (required).
    - **authorization** (string | null).
    - **connector_id** (string | null).
    - **headers** (object | null).
    - **require_approval** (string | null).
    - **server_description** (string | null).
    - **server_label** (string, required).
    - **server_url** (string, required).
    - **type** ("mcp", required).

### top_logprobs
- **Type**: integer | null
- **Min**: 0
- **Max**: 8
- **Description**: An integer between 0 and 8 specifying the number of most likely tokens to return at each token position, each with an associated log probability. "logprobs" must be set to true if this parameter is used.

### top_p
- **Type**: number | null
- **Default**: 1
- **Min (exclusive)**: 0
- **Max**: 1
- **Description**: An alternative to sampling with "temperature", called nucleus sampling, where the model considers the results of the tokens with "top_p" probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. It is generally recommended to alter this or "temperature" but not both.

### truncation
- **Type**: string | null
- **Description**: Not supported. Only maintained for compatibility reasons.

### user
- **Type**: string | null
- **Description**: A unique identifier representing your end-user, which can help xAI to monitor and detect abuse.

## Additional Notes
- This schema is derived from the Grok API documentation for chat completions.
- For compatibility with other APIs (e.g., OpenAI), some fields are included but may lead to request rejection if used.
- Ensure inputs adhere to the nested "One of:" and "All of:" structures to avoid validation errors.
- For more details on models, tools, or endpoints, refer to the official documentation at <https://docs.x.ai/docs/api-reference>.
