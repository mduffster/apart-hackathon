# Model Naming Convention Analysis: METR vs AI Models Datasets

## Summary
The AI models datasets (from Epoch AI) use different naming conventions than the METR benchmark data. Below are the key differences and matches found.

## Naming Pattern Differences

### 1. **GPT Models**
| METR Name | AI Dataset Name | Notes |
|-----------|----------------|-------|
| `gpt_4` | `GPT-4` | Hyphen vs underscore |
| `gpt_4_turbo` | `GPT-4 Turbo` | Hyphen + space vs underscore |
| `gpt_4o` | `GPT-4o` | Hyphen vs underscore |
| `gpt_3_5_turbo_instruct` | `GPT-3.5` (general) | **NOT FOUND** - No specific "turbo instruct" variant |
| `gpt_5` | **NOT FOUND** | Not in datasets (likely future/unreleased) |
| `gpt-oss-120b` | `gpt-oss-120b` | ✅ Exact match |

### 2. **Claude Models**
| METR Name | AI Dataset Name | Notes |
|-----------|----------------|-------|
| `claude_3_opus` | `Claude 3 Opus` | Space vs underscore |
| `claude_3_5_sonnet` | `Claude 3.5 Sonnet` | Space + decimal vs underscore |
| `claude_3_5_sonnet_20241022` | `Claude 3.5 Sonnet` | Version suffix removed in dataset |
| `claude_3_7_sonnet` | **NOT FOUND** | Not in datasets (future model) |
| `claude_4_opus` | `Claude Opus 4` | Different order, not found as "Claude 4 Opus" |
| `claude_4_sonnet` | `Claude Sonnet 4` | Different order |
| `claude_4_1_opus` | **NOT FOUND** | Not in datasets (future model) |
| `claude_sonnet_4_5` | **NOT FOUND** | Not in datasets (future model) |

### 3. **OpenAI O-Series**
| METR Name | AI Dataset Name | Notes |
|-----------|----------------|-------|
| `o1_preview` | `o1-preview` | Hyphen vs underscore |
| `o1_elicited` | **NOT FOUND** | Not in datasets (specialized variant) |
| `o3` | `o3` | ✅ Exact match |
| `o4-mini` | `o4-mini` | ✅ Exact match |

### 4. **DeepSeek Models**
| METR Name | AI Dataset Name | Notes |
|-----------|----------------|-------|
| `deepseek_v3` | `DeepSeek-V3` | Capitalized + hyphen vs lowercase underscore |
| `deepseek_v3_0324` | **NOT FOUND** | Version-dated variants not in dataset |
| `deepseek_r1` | `DeepSeek-R1` | ✅ Match with normalization |
| `deepseek_r1_0528` | **NOT FOUND** | Version-dated variants not in dataset |

### 5. **Qwen Models**
| METR Name | AI Dataset Name | Notes |
|-----------|----------------|-------|
| `qwen_2_72b` | `Qwen2.5-72B` or `Qwen2-72B` | Version numbers with decimals/hyphens |
| `qwen_2_5_72b` | `Qwen2.5-72B` | ✅ Match with normalization |

### 6. **Google Models**
| METR Name | AI Dataset Name | Notes |
|-----------|----------------|-------|
| `gemini_2_5_pro_preview` | `Gemini 2.5 Pro` (without "preview") | Preview suffix not in dataset |
| `davinci_002` | `GPT-3 175B (davinci)` | Different naming structure |

### 7. **Other Models**
| METR Name | AI Dataset Name | Notes |
|-----------|----------------|-------|
| `gpt2` | `GPT-2` | Hyphen vs no separator |
| `grok_4` | **NOT FOUND** | Not in datasets (future model) |

## Key Naming Convention Patterns

### METR Conventions:
1. **All lowercase** with underscores: `claude_3_opus`, `gpt_4o`
2. **Version dates as suffixes**: `claude_3_5_sonnet_20241022`
3. **Underscores for spaces**: `gemini_2_5_pro_preview`
4. **Numbers directly attached**: `gpt2`, `davinci_002`

### AI Datasets Conventions:
1. **Title Case** with proper spacing: `Claude 3 Opus`, `GPT-4 Turbo`
2. **Hyphens and spaces**: `GPT-4o`, `DeepSeek-R1`
3. **Decimal points in versions**: `Claude 3.5`, `Qwen2.5`
4. **Hyphens for separators**: `GPT-2`, `o1-preview`

## Models in METR But NOT in AI Datasets

These models are either too recent, specialized variants, or future models:

1. **gpt_3_5_turbo_instruct** - Specific instruction-tuned variant
2. **claude_3_7_sonnet** - Future/unreleased (Feb 2025 release date)
3. **claude_4_1_opus** - Future/unreleased (Aug 2025 release date)
4. **claude_sonnet_4_5** - Future/unreleased (Sep 2025 release date)
5. **gpt_5** - Future/unreleased (Aug 2025 release date)
6. **grok_4** - Future/unreleased (Jul 2025 release date)
7. **o1_elicited** - Specialized variant
8. **gemini_2_5_pro_preview** - Preview version (Apr 2025)
9. **Version-dated variants** (e.g., `deepseek_v3_0324`, `deepseek_r1_0528`, `claude_3_5_sonnet_20241022`)

## Recommendations for Joining Data

### Strategy 1: Fuzzy Matching with Normalization
1. Convert to lowercase
2. Remove underscores/hyphens
3. Remove version suffixes (dates, "preview", etc.)
4. Match on core model name

### Strategy 2: Manual Mapping Dictionary
Create explicit mappings for known models:
```python
METR_TO_AI_MAPPING = {
    'gpt_4': 'GPT-4',
    'gpt_4_turbo': 'GPT-4 Turbo',
    'gpt_4o': 'GPT-4o',
    'claude_3_opus': 'Claude 3 Opus',
    'claude_3_5_sonnet': 'Claude 3.5 Sonnet',
    'o1_preview': 'o1-preview',
    'o3': 'o3',
    'o4-mini': 'o4-mini',
    'deepseek_r1': 'DeepSeek-R1',
    'deepseek_v3': 'DeepSeek-V3',
    'qwen_2_5_72b': 'Qwen2.5-72B',
    'qwen_2_72b': 'Qwen2-72B',
    'gpt2': 'GPT-2',
    # Add more as needed
}
```

### Strategy 3: Combine Both
1. Use manual mapping for high-confidence matches
2. Fall back to fuzzy matching for others
3. Flag low-confidence matches for manual review

## Best Dataset: `all_ai_models.csv`

With **69 potential matches** (4 direct + 65 fuzzy), this dataset provides:
- Most comprehensive coverage (3,170 models)
- Best overlap with METR models
- Most columns for detailed analysis (56 columns including parameters, compute, organization, etc.)

## Next Steps

1. Create a robust matching algorithm combining strategies
2. Build a joined dataset with METR benchmarks + AI model metadata
3. Handle unmatched models explicitly
4. Validate matches manually for critical models

