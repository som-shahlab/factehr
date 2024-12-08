library(irr)
library(caret)
library(dplyr)
library(ggplot2)
library(patchwork)
library(purrr)
library(tidyr)

main_results_path <- "~/Documents/radgraph/final_results.csv"
nli_path <- "~/Documents/radgraph/nli_results/"
annotations_path <- "~/Documents/radgraph/annotators/"
annotation_exercise_path <- "~/Documents/radgraph/full_annotation_task.csv"

get_classification_metrics <- function(df, truth, pred) {
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

# main results
in_df <- readr::read_csv(main_results_path) %>% 
  filter(model != "entailment_results",
         dataset != "entailment_results_final") %>% 
  mutate(icl = grepl("ICL", prompt)) %>% 
  select(id = ID, precision, recall, f1, model, dataset, prompt, note, icl) %>% 
  mutate(prompt = gsub("^[a-z]*_", "", prompt),
         note = case_when(note %in% c("breastca", "pdac") ~ "progress_note",
                          T ~ note))

# in_df %>% 
#   count(model, dataset, prompt, note) %>% 
#   pivot_wider(id_cols = c(model, prompt, dataset),
#               names_from = note,
#               values_from = n) %>% 
#   View()

all_raw_results <- in_df %>% 
  group_by(dataset, model, note, prompt) %>% 
  summarize(n_reports = n(),
            f1 = mean(f1),
            precision = mean(precision),
            recall = mean(recall)) %>% 
  arrange(note, desc(f1)) %>% 
  select(note,dataset, model, prompt, everything()) %>% 
  mutate(f1 = 2*precision*recall/(precision + recall)) %>% 
  mutate_at(vars(f1, precision, recall), ~round(.x * 100, 1)) %>% 
  ungroup()


# ablation_results
group_by_list <- list(by_icl = c("icl"),
                      by_model_icl = c("model", "icl"),
                      by_prompt = c("prompt"),
                      by_model = c("model"))


ablation_results <- map(group_by_list,
    ~{group_vars <- syms(c(.x, "dataset"))
    in_df %>% 
      group_by(!!!group_vars) %>% 
      summarize_at(vars(f1, precision, recall),
                   ~glue::glue("{round(mean(.x, na.rm=T) * 100, 1)} ({round(sd(.x, na.rm=T) * 100, 0)})")) %>% 
      arrange(dataset) %>% 
      select(dataset, everything())})

ablation_results_plot_dfs <- map(group_by_list,
                        ~{group_vars <- syms(c(.x, "note"))
                        in_df %>% 
                          group_by(!!!group_vars) %>% 
                          summarize_at(vars(f1, precision, recall),
                                       ~mean(.x * 100, na.rm = T)) %>% 
                          arrange(note) %>% 
                          select(note, everything())})

add_theme <- function(gplot) {
  gplot + 
    ylab("F1") +
    xlab("")
}

by_icl_plot <- ablation_results_plot_dfs$by_icl %>% 
  ggplot(aes(x = note, y = f1, fill = icl)) +
  geom_bar(position = "dodge", stat = "identity")  +
  xlab("")

by_model_icl_plot <- ablation_results_plot_dfs$by_model_icl %>% 
  ungroup() %>% 
  ggplot(aes(x = note, y = f1, fill = icl)) +
  geom_bar(position = "dodge", stat = "identity")+
    facet_wrap(~model)  +
  xlab("")

by_prompt_plot <- ablation_results_plot_dfs$by_prompt %>% 
  ungroup() %>% 
  ggplot(aes(x = note, y = f1, fill = prompt)) +
  geom_bar(position = "dodge", stat = "identity")  +
  xlab("")

by_model_plot <- ablation_results_plot_dfs$by_model %>% 
  ungroup() %>% 
  ggplot(aes(x = note, y = f1, fill = model)) +
  geom_bar(position = "dodge", stat = "identity")  +
  xlab("")

(by_model_plot + by_prompt_plot) / by_model_icl_plot

(by_model_plot + by_prompt_plot + by_icl_plot)
(by_model_plot / by_prompt_plot / by_icl_plot)

# med NLI
data_paths <- paste0(nli_path, list.files(nli_path)) %>% 
  setNames(gsub("\\.csv", "", list.files(nli_path))) %>% 
  as.list()

# test <- readr::read_csv(data_paths$`multi_nli_results_deberta-large-mnli`)

get_nli_results <- function(med_nli_path) {
  mednli_df <- data.table::fread(med_nli_path, header = T) %>% 
    tibble() %>% 
    mutate(
      pred = substr(pred_label, 2, 2) %>% as.numeric(),
      gold_label = case_when(gold_label == 0 ~ "entailment",
                              T ~ as.character(gold_label)),
      truth = case_when(
        gold_label %in% c("entailment") ~ 1,
        TRUE ~ 0
      )
    )
  
  return(mednli_df)

}

raw_results <- imap_dfr(data_paths, ~get_nli_results(.x) %>% 
                      mutate(model = strsplit(.y, "results")[[1]][2], 
                             dataset = strsplit(.y, "results")[[1]][1]) %>% 
                      mutate_if(is.numeric, round, 3)) %>% 
  mutate(dataset = case_when(dataset == "mli_test_v1_" ~ "mednli",
                             T ~ dataset),
         model = gsub("^_", "", model),
         dataset = gsub("_$", "", dataset))

model_results <- raw_results %>% 
  group_by(model) %>% 
  nest() %>% 
  mutate(tbl = map(data, get_classification_metrics, pred = "pred", truth = "truth")) %>% 
  select(model, tbl) %>% 
  unnest(tbl) %>% 
  filter(model != "MedLM-large") %>% 
  arrange(f1)

model_dataset_results <-  raw_results %>% 
  group_by(model, dataset) %>% 
  nest() %>% 
  mutate(tbl = map(data, get_classification_metrics, pred = "pred", truth = "truth")) %>% 
  select(model, dataset, tbl) %>% 
  unnest(tbl) %>% 
  filter(model != "MedLM-large")

make_nli_plot <- function(in_df) {
  in_df %>% 
    pivot_longer(cols = c(sensitivity, ppv, f1, accuracy),
                 names_to = "metric",
                 values_to = "value") %>% 
    ggplot(aes(x = model, y = value, fill = model)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme(axis.text.x = element_blank()) +
    xlab("") +
    ylab("")
}

make_nli_plot(model_results) + 
  facet_wrap(~metric, ncol = 4) 

make_nli_plot(model_dataset_results) + 
  facet_wrap(~dataset+metric, ncol = 4) 

# annotator agreement
annotator_file_names <- paste0(annotations_path, list.files(annotations_path))

annotator_dfs <- map(annotator_file_names, readxl::read_xlsx) %>% 
  setNames(list.files(annotations_path))

cleaned_df <- annotator_dfs %>% 
  imap_dfr(~.x %>% 
         mutate(annotator = .y)) %>% 
  filter(!is.na(inferrable)) %>% 
  filter(nchar(claim) > 10)

ratings <- cleaned_df %>% 
  filter(row_number %in% 1:100) %>% 
  pivot_wider(id_cols = c(row_number),
              names_from = annotator,
              values_from = inferrable) %>% 
  select(-row_number) %>% 
  as.matrix()

fleiss_kappa <- kappam.fleiss(ratings, exact = FALSE, detail = FALSE) %>% 
  map(~.x) %>% 
  data.frame()

# annotation of entailment
annotator_task_preds <- readr::read_csv(annotation_exercise_path)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

annotation_final <- annotator_task_preds %>% 
  select(row_number, entailment_pred) %>% 
  right_join(cleaned_df, by = "row_number") %>% 
  filter(!is.na(entailment_pred),
         !is.na(inferrable)) %>% 
  group_by(row_number, entailment_pred) %>% 
  summarize(truth = Mode(inferrable)) %>% 
  ungroup()

library(jsonlite)
out_jsonl <- annotation_final %>% 
  left_join(annotator_task_preds) %>% 
  transmute(row_number,
            sentence1 = premise,
         sentence2 = hypothesis,
         gold_label = case_when(truth == 1 ~ "entailment",
                                T ~ "contradiction"))
jsonl_file <- "~/Documents/radgraph/clinician_annotations_old.jsonl"

# Open a connection to the JSONL file
con <- file(jsonl_file, open = "wt")

# Write each row as a separate JSON line
for (i in 1:nrow(out_jsonl)) {
  json_line <- toJSON(out_jsonl[i, ], auto_unbox = TRUE)
  json_line <- gsub("^\\[|\\]$", "", json_line)  # Remove the square brackets
  writeLines(json_line, con)
}

# Close the connection
close(con)



entailment_validation <- get_classification_metrics(annotation_final, truth = "truth",
                           pred = "entailment_pred")


# collate and write to google sheet

out <- list("all_results_raw" = all_raw_results) %>% 
  append(ablation_results) #%>%
  # append(list(med_nli = med_nli_results,
  #             annotator_agreement = fleiss_kappa,
  #             entailment_validation = entailment_validation
# ))

imap(out,
     ~googlesheets4::write_sheet(.x, 
                                 sheet = .y,
                                 # ss = "https://docs.google.com/spreadsheets/d/15N4_Hiblvpph9We7PUYs1t37MWvx6cfrihboKyN2a4Y/edit#gid=0",
                                 ss = "https://docs.google.com/spreadsheets/d/1q2LPJ5c576_Jfc6pXFwMUEuSuD8YXV4AKHQku131nnI/edit?gid=0#gid=0"
                                 )
)




# error annotation
# error_df <- readr::read_csv("~/Documents/radgraph/error_annotation_raw.csv")
# 
# error_df %>% 
#   mutate(split = str_split(key, "_"),
#          precision_recall = map_chr(split, tail, 1),
#          model = map_chr(split, head, 1),
#          dataset = map_chr(split, ~.x[2]),
#          prompt = map_chr(split, ~.x[3])) %>% 
#   group_by(model, dataset, prompt, precision_recall) %>% 
#   sample_n(pmin(30, n())) %>% View #write_excel_csv("~/Documents/radgraph/error_annotation_raw_sampled.csv")
#   data.frame()

