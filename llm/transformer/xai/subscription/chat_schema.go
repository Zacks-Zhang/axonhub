package subscription

import (
	"encoding/json"
	"fmt"
	"reflect"
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

		inlined := make([]any, 0, len(branches))
		for _, branch := range branches {
			resolved, err := inlineLocalRefs(branch, definitions, nil)
			if err != nil {
				return err
			}

			inlined = append(inlined, resolved)
		}

		normalized, ok := flattenSoleUnionBranches(inlined, unionKey)
		if !ok {
			normalized = inlined
		}
		schema[unionKey] = normalized
	}

	return nil
}

func flattenSoleUnionBranches(branches []any, unionKey string) ([]any, bool) {
	normalized := make([]any, 0, len(branches))
	changed := false

	for _, branch := range branches {
		nested, ok := soleUnionBranches(branch, unionKey)
		if !ok {
			normalized = append(normalized, branch)
			continue
		}

		normalized = append(normalized, nested...)
		changed = true
	}

	if !changed {
		return branches, false
	}
	if unionKey == "oneOf" && !branchesAreMutuallyExclusive(normalized) {
		return nil, false
	}

	return normalized, true
}

func soleUnionBranches(value any, unionKey string) ([]any, bool) {
	schema, ok := value.(map[string]any)
	if !ok || len(schema) != 1 {
		return nil, false
	}

	branches, ok := schema[unionKey].([]any)
	if !ok || len(branches) == 0 {
		return nil, false
	}

	return branches, true
}

func branchesAreMutuallyExclusive(branches []any) bool {
	for i := 0; i < len(branches); i++ {
		left, ok := branches[i].(map[string]any)
		if !ok {
			return false
		}

		for j := i + 1; j < len(branches); j++ {
			right, ok := branches[j].(map[string]any)
			if !ok {
				return false
			}
			if !branchesAreDisjoint(left, right) {
				return false
			}
		}
	}

	return true
}

func branchesAreDisjoint(left, right map[string]any) bool {
	leftProperties, _ := left["properties"].(map[string]any)
	rightProperties, _ := right["properties"].(map[string]any)
	rightRequired := requiredProperties(right)

	for required := range requiredProperties(left) {
		if _, ok := rightRequired[required]; !ok {
			continue
		}

		leftValues, ok := enumValues(leftProperties[required])
		if !ok {
			continue
		}
		rightValues, ok := enumValues(rightProperties[required])
		if !ok {
			continue
		}
		if valuesAreDisjoint(leftValues, rightValues) {
			return true
		}
	}

	return false
}

func requiredProperties(schema map[string]any) map[string]struct{} {
	required := make(map[string]struct{})
	values, ok := schema["required"].([]any)
	if !ok {
		return required
	}

	for _, value := range values {
		if name, ok := value.(string); ok {
			required[name] = struct{}{}
		}
	}

	return required
}

func enumValues(schema any) ([]any, bool) {
	properties, ok := schema.(map[string]any)
	if !ok {
		return nil, false
	}

	if values, ok := properties["enum"].([]any); ok {
		return values, true
	}
	if value, ok := properties["const"]; ok {
		return []any{value}, true
	}

	return nil, false
}

func valuesAreDisjoint(left, right []any) bool {
	for _, leftValue := range left {
		for _, rightValue := range right {
			if reflect.DeepEqual(leftValue, rightValue) {
				return false
			}
		}
	}

	return true
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
