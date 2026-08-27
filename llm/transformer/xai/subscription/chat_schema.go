package subscription

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/transformer"
)

// normalizeChatRequestToolSchemas expands local $ref branches in root-level
// oneOf/anyOf function schemas. The Grok subscription Chat endpoint rejects
// those references even though they are valid JSON Schema. Normalization is
// deliberately scoped to the xAI subscription Chat outbound.
func normalizeChatRequestToolSchemas(request *llm.Request) (*llm.Request, error) {
	if request == nil || len(request.Tools) == 0 || !llm.IsOpenAIResponsesFormat(request.APIFormat) {
		return request, nil
	}

	tools := append([]llm.Tool(nil), request.Tools...)
	changed := false

	for i, tool := range tools {
		if tool.Type != llm.ToolTypeFunction || len(tool.Function.Parameters) == 0 {
			continue
		}

		parameters, err := normalizeChatFunctionParameters(tool.Function.Parameters)
		if err != nil {
			return nil, fmt.Errorf("%w: function %q parameters: %v", transformer.ErrInvalidRequest, tool.Function.Name, err)
		}

		if string(parameters) != string(tool.Function.Parameters) {
			tool.Function.Parameters = parameters
			tools[i] = tool
			changed = true
		}
	}

	if !changed {
		return request, nil
	}

	normalized := *request
	normalized.Tools = tools

	return &normalized, nil
}

func normalizeChatFunctionParameters(parameters json.RawMessage) (json.RawMessage, error) {
	var schema map[string]any
	if err := json.Unmarshal(parameters, &schema); err != nil || !hasRootUnionRefBranch(schema) {
		return parameters, nil
	}

	if err := inlineRootUnionRefBranches(schema); err != nil {
		return nil, err
	}

	normalized, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to normalize parameters schema: %w", err)
	}

	return normalized, nil
}

func hasRootUnionRefBranch(schema map[string]any) bool {
	for _, unionKey := range []string{"oneOf", "anyOf"} {
		branches, ok := schema[unionKey].([]any)
		if !ok {
			continue
		}

		for _, branch := range branches {
			branchMap, ok := branch.(map[string]any)
			if !ok {
				continue
			}

			ref, ok := branchMap["$ref"].(string)
			if ok && isLocalDefinitionRef(ref) {
				return true
			}
		}
	}

	return false
}

func inlineRootUnionRefBranches(schema map[string]any) error {
	definitions, _ := schema["$defs"].(map[string]any)

	for _, unionKey := range []string{"oneOf", "anyOf"} {
		branches, ok := schema[unionKey].([]any)
		if !ok {
			continue
		}

		for i, branch := range branches {
			inlined, err := inlineLocalRefs(branch, definitions, nil)
			if err != nil {
				return err
			}

			branches[i] = inlined
		}
	}

	return nil
}

func inlineLocalRefs(value any, definitions map[string]any, active map[string]struct{}) (any, error) {
	switch typed := value.(type) {
	case map[string]any:
		ref, ok := typed["$ref"].(string)
		if !ok {
			result := make(map[string]any, len(typed))
			for key, nested := range typed {
				inlined, err := inlineLocalRefs(nested, definitions, active)
				if err != nil {
					return nil, err
				}
				result[key] = inlined
			}
			return result, nil
		}

		name, ok := localDefinitionName(ref)
		if !ok {
			return typed, nil
		}

		definition, ok := definitions[name]
		if !ok {
			return nil, fmt.Errorf("local definition %q is missing", name)
		}

		if active == nil {
			active = make(map[string]struct{})
		}
		if _, ok := active[name]; ok {
			return nil, fmt.Errorf("local definition %q is cyclic", name)
		}

		nextActive := make(map[string]struct{}, len(active)+1)
		for key := range active {
			nextActive[key] = struct{}{}
		}
		nextActive[name] = struct{}{}

		resolved, err := inlineLocalRefs(definition, definitions, nextActive)
		if err != nil {
			return nil, err
		}
		if resolvedMap, ok := resolved.(map[string]any); ok {
			for key, sibling := range typed {
				if key == "$ref" {
					continue
				}
				resolvedMap[key] = sibling
			}
			return resolvedMap, nil
		}

		return resolved, nil
	case []any:
		result := make([]any, len(typed))
		for i, nested := range typed {
			inlined, err := inlineLocalRefs(nested, definitions, active)
			if err != nil {
				return nil, err
			}
			result[i] = inlined
		}
		return result, nil
	default:
		return typed, nil
	}
}

func isLocalDefinitionRef(ref string) bool {
	_, ok := localDefinitionName(ref)
	return ok
}

func localDefinitionName(ref string) (string, bool) {
	const prefix = "#/$defs/"
	if !strings.HasPrefix(ref, prefix) {
		return "", false
	}

	name := strings.TrimPrefix(ref, prefix)
	if name == "" || strings.ContainsAny(name, "/~") {
		return "", false
	}

	return name, true
}
