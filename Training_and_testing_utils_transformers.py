import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    precision_recall_fscore_support
)

from neural_text_models import *
from Utils import save_data


def train_model(location, model_type, X_train, att_mask_train, y_train,
                X_valid, att_mask_valid, y_valid, device, batch_size,
                accumulation_steps, num_epochs, num_classes, report_every,
                epoch_patience, load=False, pretrain=False, noisy_student=False,
                learning_rate=2e-5):

    x_tr = torch.tensor(X_train, dtype=torch.long)
    att_mask_tr = torch.tensor(att_mask_train, dtype=torch.long)
    y_tr = torch.tensor(y_train, dtype=torch.long)

    x_val = torch.tensor(X_valid, dtype=torch.long)
    att_mask_val = torch.tensor(att_mask_valid, dtype=torch.long)
    y_val = torch.tensor(y_valid, dtype=torch.long)

    train_dataset = TensorDataset(x_tr, y_tr, att_mask_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(x_val, y_val, att_mask_val)
    validation_loader = DataLoader(val_dataset, batch_size=64)

    # Model selection
    if model_type == "Sci_BERT":
        model = Sci_BERT(num_classes)
    elif model_type == "Bio_BERT":
        model = Bio_BERT(num_classes)
    elif model_type == 'Sci_BERT_uncased':
        model = Sci_BERT_uncased(num_classes)
    elif model_type == "RoBERTa":
        model = RoBERTa(num_classes)
    elif model_type == "RoBERTa_large":
        model = RoBERTa_large(num_classes)
    elif model_type == "RoBERTa_large_for_contrastive":
        model = RoBERTa_large_for_contrastive(num_classes)
    elif model_type == "xlnet":
        model = XLnet(num_classes)
    elif model_type == "BERT_local":
        # Ensure pretrained_model_loc is defined if this option is used
        model = BERT_locally_pretrained(num_classes, pretrained_model_loc)
    elif model_type == "BERT_contrastive":
        model = BERT_for_contrastive(num_classes)
    else: # Default to BERT
        model = BERT(num_classes)
        if noisy_student:
            model = BERT_with_dropout(num_classes, 0.5)

    print(f"Model loaded: {model_type}")
    model.to(torch.device(device))

    if load:
        model.load_state_dict(torch.load(location + '/model.pt'))
        print("Loaded pretrained model weights.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    prev_best_score = -1.0
    epochs_no_improve = 0
    best_performing_epoch = 0

    for epoch in range(num_epochs):
        if epochs_no_improve == epoch_patience:
            print(f"Validation performance not improving for {epoch_patience} consecutive epochs. Stopping training.")
            break

        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for i, (data, target, att) in enumerate(train_loader):
            data, target, att = data.to(device), target.to(device), att.to(device)

            if model_type in ["BERT_contrastive", "RoBERTa_large_for_contrastive"]:
                output, _ = model(data, att)
            else:
                output = model(data, att)

            loss = criterion(output, target) / accumulation_steps
            loss.backward()
            running_loss += loss.item() * accumulation_steps # Accumulate actual loss

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Validation phase
        model.eval()
        val_predictions = []
        with torch.no_grad():
            for val_data, val_target, att_val in validation_loader:
                val_data, att_val = val_data.to(device), att_val.to(device)
                if model_type in ["BERT_contrastive", "RoBERTa_large_for_contrastive"]:
                    out, _ = model(val_data, att_val)
                else:
                    out = model(val_data, att_val)
                val_predictions.extend(torch.argmax(out, dim=1).cpu().tolist())

        current_score = f1_score(y_valid.tolist(), val_predictions, average="macro", zero_division=0)
        print(f"Epoch {epoch}: Validation F1 Macro: {current_score:.4f}")

        if current_score > prev_best_score:
            best_performing_epoch = epoch
            print(f"Validation F1 score improved from {prev_best_score:.4f} to {current_score:.4f}. Saving model...")
            prev_best_score = current_score
            checkpoint_info = {'epoch': epoch, 'score': prev_best_score}
            save_data(checkpoint_info, location + '/checkpoint.pkl')
            torch.save(model.state_dict(), location + '/model.pt')
            with open(location + '/best_epoch.txt', 'w') as f:
                f.write(f'best epoch: {best_performing_epoch}, f1_macro: {current_score:.4f}')
            epochs_no_improve = 0
        else:
            print(f"Validation F1 score ({current_score:.4f}) did not improve from {prev_best_score:.4f}.")
            epochs_no_improve += 1
    print(f"Finished training. Best model from epoch {best_performing_epoch} with F1 Macro: {prev_best_score:.4f}")


def test_model(location, model_type, x_test_data, att_x_test, y_test_labels, device,
               batch_size, num_classes, print_res, save_loc, test_df=None):

    if model_type == "Sci_BERT":
        trained_model = Sci_BERT(num_classes)
    elif model_type == 'Sci_BERT_uncased':
        trained_model = Sci_BERT_uncased(num_classes)
    elif model_type == "Bio_BERT":
        trained_model = Bio_BERT(num_classes)
    elif model_type == "RoBERTa":
        trained_model = RoBERTa(num_classes)
    elif model_type == "RoBERTa_large":
        trained_model = RoBERTa_large(num_classes)
    elif model_type == "xlnet":
        trained_model = XLnet(num_classes)
    elif model_type == "BERT_local":
        # Ensure pretrained_model_loc is defined
        trained_model = BERT_locally_pretrained(num_classes, pretrained_model_loc)
    elif model_type == "BERT_contrastive":
        trained_model = BERT_for_contrastive(num_classes)
    elif model_type == 'RoBERTa_large_for_contrastive':
        trained_model = RoBERTa_large_for_contrastive(num_classes)
    else: # Default to BERT
        trained_model = BERT(num_classes)

    trained_model.to(torch.device(device))
    # Load the best saved model weights
    model_path = location + '/model.pt'
    try:
        trained_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        print(f"Loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Please ensure training was successful and path is correct.")
        return None, None


    x_te = torch.tensor(x_test_data, dtype=torch.long)
    att_mask_te = torch.tensor(att_x_test, dtype=torch.long)

    test_dataset = TensorDataset(x_te, att_mask_te)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    trained_model.eval()
    test_predictions = []
    all_probabilities_list = []

    with torch.no_grad():
        for test_data_batch, att_batch in test_loader:
            test_data_batch, att_batch = test_data_batch.to(device), att_batch.to(device)

            if model_type in ['BERT_contrastive', 'RoBERTa_large_for_contrastive']:
                out, _ = trained_model(test_data_batch, att_batch)
            else:
                out = trained_model(test_data_batch, att_batch)

            probabilities = torch.softmax(out, dim=1)
            predictions_batch = torch.argmax(out, dim=1)

            test_predictions.extend(predictions_batch.cpu().tolist())
            all_probabilities_list.extend(probabilities.cpu().tolist())

    y_true_list = y_test_labels.tolist() if isinstance(y_test_labels, np.ndarray) else y_test_labels
    
    save_data(test_predictions, location + '/test_output.pkl')
    save_data(all_probabilities_list, location + '/all_probabilities.pkl')

    # Calculate overall metrics
    f1_macro_overall = f1_score(y_true_list, test_predictions, average="macro", zero_division=0)
    accuracy_overall = accuracy_score(y_true_list, test_predictions)

    print(f"\nOverall Test Metrics:")
    print(f"Macro F1-score: {f1_macro_overall:.4f}")
    print(f"Accuracy: {accuracy_overall:.4f}")

    target_names_map = {
        3: ['entailment', 'neutral', 'contradiction'],
        4: ['contrasting', 'reasoning', 'entailing', 'neutral']
    }
    report_target_names = target_names_map.get(num_classes, [f"class_{i}" for i in range(num_classes)])

    if print_res: # print_res flag from function arguments
        print("\nClassification Report (Overall):")
        print(classification_report(y_true_list, test_predictions, target_names=report_target_names, digits=4, zero_division=0))

    performance_dict = classification_report(y_true_list, test_predictions, target_names=report_target_names, digits=4, output_dict=True, zero_division=0)
    performance_df = pandas.DataFrame(performance_dict).transpose()

    # Domain-specific evaluation
    if test_df is not None and 'Domain' in test_df.columns:
        domain_metrics = defaultdict(dict)
        unique_domains = sorted(list(set(test_df['Domain'])))

        for domain in unique_domains:
            domain_indices = test_df[test_df['Domain'] == domain].index
            # Ensure indices are valid and map correctly to y_true_list and test_predictions
            domain_true = [y_true_list[i] for i in domain_indices if i < len(y_true_list)]
            domain_pred = [test_predictions[i] for i in domain_indices if i < len(test_predictions)]
            
            if not domain_true or not domain_pred: # Skip if no data for domain after filtering
                print(f"Skipping domain '{domain}' due to insufficient data after index mapping.")
                continue

            acc = accuracy_score(domain_true, domain_pred)
            prec_macro, rec_macro, f1_mac, _ = precision_recall_fscore_support(
                domain_true, domain_pred, average='macro', zero_division=0
            )
            domain_metrics[domain] = {
                "Accuracy": acc, "F1_Macro": f1_mac,
                "Precision_Macro": prec_macro, "Recall_Macro": rec_macro
            }

        domain_metrics["Overall"] = {
            "Accuracy": accuracy_overall, "F1_Macro": f1_macro_overall,
            "Precision_Macro": performance_dict['macro avg']['precision'],
            "Recall_Macro": performance_dict['macro avg']['recall']
        }
        metrics_df = pandas.DataFrame(domain_metrics).T
        print("\nDomain-Specific Classification Report:")
        print(metrics_df)
        metrics_df.to_csv(location + '/domain_metrics.csv')
        # Save combined report
        performance_df.to_csv(save_loc.replace(".csv", "_overall.csv"))
        metrics_df.to_csv(save_loc.replace(".csv", "_domain.csv"))
        print(f"Overall report saved to {save_loc.replace('.csv', '_overall.csv')}")
        print(f"Domain-specific report saved to {save_loc.replace('.csv', '_domain.csv')}")
    else:
        performance_df.to_csv(save_loc)
        print(f"Overall performance report saved to {save_loc}")
        
    # Calculate max probabilities for predicted class
    all_max_probabilities = [probs[pred_idx] for probs, pred_idx in zip(all_probabilities_list, test_predictions)]

    return test_predictions, all_max_probabilities