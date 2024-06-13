# What is a Causal Language Model?

A causal language model is a type of language model that predicts the next token in a sequence of tokens. The model generates text one token at a time, with each token conditioned on the tokens that came before it. This type of model is useful for generating text, such as in chatbots, machine translation, and text summarization. [see more](https://huggingface.co/docs/transformers/tasks/language_modeling)


# The Causal Language Model Contract
In the remaining sections, we will describe the contract for a causal language model.

## `CausalLMModelInput`
```C#
public CausalLMModelInput
{
    // [batch_size, sequence_length]
    public Tensor input_ids { get; set; }

    // optional: [batch_size, sequence_length]
    public Tensor? attention_mask { get; set; }

    // optional: [batch_size, sequence_length]
    public Tensor? position_ids { get; set; }

    // optional: kv cache for attention layers
    public IKVCache? kv_cache { get; set; }

    // optional: [batch_size, sequence_length, hidden_size]
    // if provided, the model will use these embeddings instead of computing them from input_ids
    public Tensor? inputs_embeds { get; set; }

    // if use kv cache when calculating attention
    public bool use_cache { get; set; }

    // if return attentions in model output
    public bool output_attentions { get; set; }

    // if return hidden states in model output
    // for e.g. calculating loss
    public bool output_hidden_states { get; set; }
}
```

## `CausalLMModelOutput`
```C#
public class CausalLMModelOutput
{
    // [batch_size, sequence_length, vocab_size]
    // The predicted logits for each token in the input sequence.
    public Tensor logits { get; set; }

    // optional: [batch_size, sequence_length, hidden_size]
    public Tensor last_hidden_state { get; set; }

    // optional: all hidden states
    public Tensor[]? hidden_states { get; set; }

    // optional: all attentions
    public Tensor[]? attentions { get; set; }

    // optional: kv cache for attention layers
    public IKVCache? cache { get; set; }
}
```

Once both `CausalLMModelInput` and `CausalLMModelOutput` are defined, the causal language model can be implemented as follows (use Phi-3 as an example):

```C#
public class Phi3ForCausalLM : nn.Module<CausalLMModelInput, CausalLMModelOutput>
```


# What language model has been implemented using this contract in this repo?
- `Phi3ForCausalLM`
- `Phi2ForCausalLM`

# What language model has been implemented using this pattern, but not exactly the same contract class in the other repo?
- `LLaMAForCausalLM` (for both llama2 and llama3)
