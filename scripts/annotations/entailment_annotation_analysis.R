# Entailment annotation analysis
library(irr)
library(caret)
library(dplyr)
library(ggplot2)
library(patchwork)
library(purrr)
library(tidyr)

get_classification_metrics <- function(df, truth, pred) {
  # Get classification metrics given a dataframe (df), truth column name and prediction column name (strings)
  conf_matrix <- confusionMatrix(
    factor(df[[pred]], levels = c(1, 0)),
    factor(df[[truth]], levels = c(1, 0))
  )
  
  # Extract metrics
  sensitivity <- conf_matrix$byClass['Sensitivity']
  specificity <- conf_matrix$byClass['Specificity']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  npv <- conf_matrix$byClass['Neg Pred Value']
  f1 <- conf_matrix$byClass['F1']
  accuracy <- conf_matrix$overall['Accuracy']
  
  # Print the results
  out <- data.frame(
    sensitivity = sensitivity,
    specificity = specificity,
    ppv = ppv,
    npv = npv,
    f1 = f1,
    accuracy = accuracy
  ) %>% 
    `rownames<-`(NULL)
  
  return(out)
}

final_df <- readr::read_csv("~/Documents/radgraph/annotation_tasks/human_annotations_merged_120824.csv") %>% 
  select(-...1)
base_annotation_file <- readr::read_csv("~/Downloads/annotation_file_101724.csv") %>% 
  mutate(unique_id = row_number())
entailment_pairs <- readr::read_csv("/Users/akshayswaminathan/Downloads/release_files_final/entailment_pairs_110424.csv") %>% 
  select(-...1)
medalign_df <- readr::read_csv("~/Documents/radgraph/annotation_tasks/medalign_annotations_101824.csv")
mehr_medalign_annotations <- readxl::read_xlsx("~/Documents/radgraph/annotation_task_completed_11_2024/entailment_annotation_101824_MK_2.xlsx") %>% 
  transmute(unique_id,
            human_pred = coalesce(inferrable, 0))

annotation_tasks <- bind_rows(base_annotation_file) #bind_rows(inter_rater_agreement, individual_annotation)

entailment_pairs %>% 
  filter(doc_id == "9db2398c73f29f69ebf381f22b121e86")

# Non-medalign annotation tasks
non_medalign <- final_df %>% 
  left_join(base_annotation_file, "unique_id")

# Medalign annotations
medalign <- mehr_medalign_annotations %>% 
  left_join(medalign_df, "unique_id")

all_human_annotations <- bind_rows(non_medalign, medalign) %>% 
  select(doc_id, dataset_name, note_type, prompt, 
         index, entailment_type, model_name, premise, hypothesis, human_pred)

all_human_model_annotations <- all_human_annotations %>% 
  # left_join(annotation_tasks, "unique_id") %>% 
  left_join(entailment_pairs) %>% 
  filter(!is.na(entailment_pred),
         !is.na(human_pred)) %>% 
  select(doc_id, dataset_name, note_type, prompt, index, entailment_type, model_name, premise, hypothesis, human_pred, entailment_pred)

all_human_model_annotations %>% 
  write.csv("~/Downloads/all_human_model_entailment_labels_120824.csv")

all_human_model_annotations %>% 
  get_classification_metrics("human_pred", "entailment_pred") %>% 
  mutate(n = nrow(all_human_model_annotations)) %>% 
  write.csv("~/Downloads/factehr_entailment_human_evaluation_120824.csv")

# counts <- final_df %>% 
#   mutate(key_split = str_split(key, "\\+"),
#          model = map_chr(key_split, ~.x[[1]]),
#          dataset = map_chr(key_split, ~.x[[2]]),
#          note_type = map_chr(key_split, ~.x[[3]]),
#          prompt = map_chr(key_split, ~.x[[4]]),
#          metric = map_chr(key_split, ~.x[[5]])) %>% 
#   select(model, dataset, note_type, prompt, metric)
# 
# map(names(counts),
#     ~counts %>% 
#       count(!!sym(.x), sort = T) %>% 
#       mutate(perc = n / sum(n))) %>% 
#   setNames(names(counts)) %>% 
#   imap(~write_csv(.x, file = glue::glue("/tmp/FactEHR_counts_{.y}.csv")))



