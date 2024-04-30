from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
class CustomBertModel(BertPreTrainedModel):
    def __init__(self, config, layer_num=-1, num_labels=2):  # Defaulting num_labels to 2 for binary classification
        super().__init__(config)
        config.return_dict = True
        self.bert = BertModel(config)
        self.layer_num = layer_num
        self.num_labels = num_labels  # Define num_labels in your model
        self.classifier = nn.Linear(config.hidden_size, num_labels)  # Adjust based on the number of labels

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        
        # Select the embeddings from the specified layer
        sequence_output = outputs.hidden_states[self.layer_num]
        logits = self.classifier(sequence_output[:, 0, :])  # We take the embedding of the first token ([CLS]) for classification.

        # Continue with the loss calculation if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # print("Logits Shape and Sample:", logits.shape, logits[:2])  # Show shape and first two entries
        # print("Hidden States Example:", outputs.hidden_states[-1].shape if outputs.hidden_states else "No hidden states")  # Last layer's shape
        # print("Attentions Example:", outputs.attentions[-1].shape if outputs.attentions else "No attentions")  # Last attention layer's shape


        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        )
