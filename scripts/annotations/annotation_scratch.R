library(dplyr)
library(ggplot2)

in_path <- "/Users/akshayswaminathan/just-the-facts/data/datasets/output/metrics_by_group.csv"

in_df <- readr::read_csv(in_path)

in_df %>% 
  tidyr::pivot_longer(cols = c(not_parseable, f1_score, precision, recall),
                      names_to = "variable",
                      values_to = "value") %>% 
  ggplot(aes(x = gsub("entailment_", "", variable), y = value, fill = gsub("entailment_", "", prompt))) +
  geom_bar(position = "dodge", stat = "identity") +
  facet_wrap(~dataset + model) +
  theme(axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   size = 12),
        strip.text = element_text(size = 15),
        legend.position = "top",
        legend.title = element_blank()) +
  xlab("") +
  ylab("")

in_df %>% 
  filter(prompt == "entailment_binary6") %>% 
  tidyr::pivot_longer(cols = c(not_parseable, f1_score, precision, recall),
                      names_to = "variable",
                      values_to = "value") %>% 
  ggplot(aes(x = gsub("entailment_", "", variable), y = value, fill = gsub("entailment_", "", model))) +
  geom_bar(position = "dodge", stat = "identity") +
  facet_wrap(~dataset) +
  theme(axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   size = 12),
        strip.text = element_text(size = 15),
        legend.position = "top",
        legend.title = element_blank()) +
  xlab("") +
  ylab("")

in_df %>% 
  group_by(dataset, prompt) %>% 
  summarise(n_total = sum(n),
            not_parseable = sum(n * not_parseable) / sum(n),
            accuracy = sum(n * accuracy) / sum(n),
            precision = sum(n * precision) / sum(n),
            recall = sum(n * recall) / sum(n),
            f1_score = sum(n * f1_score) / sum(n))


in_df %>% 
  group_by(model) %>% 
  summarise(n_total = sum(n),
            not_parseable = sum(n * not_parseable) / sum(n),
            accuracy = sum(n * accuracy) / sum(n),
            precision = sum(n * precision) / sum(n),
            recall = sum(n * recall) / sum(n),
            f1_score = sum(n * f1_score) / sum(n))

in_df %>% 
  group_by(model, dataset) %>% 
  summarise(n_total = sum(n),
            not_parseable = sum(n * not_parseable) / sum(n),
            accuracy = sum(n * accuracy) / sum(n),
            precision = sum(n * precision) / sum(n),
            recall = sum(n * recall) / sum(n),
            f1_score = sum(n * f1_score) / sum(n)) %>% 
  tidyr::pivot_longer(cols = c(not_parseable, f1_score, precision, recall),
                      names_to = "variable",
                      values_to = "value") %>% 
  ggplot(aes(x = gsub("entailment_", "", variable), y = value, fill = gsub("entailment_", "", model))) +
  geom_bar(position = "dodge", stat = "identity") +
  facet_wrap(~dataset) +
  theme(axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   size = 12),
        strip.text = element_text(size = 15),
        legend.position = "top",
        legend.title = element_blank()) +
  xlab("") +
  ylab("")

in_df %>% 
  filter(!grepl("llama", tolower(model))) %>% 
  group_by(dataset, prompt) %>% 
  summarise(n_total = sum(n),
            not_parseable = sum(n * not_parseable) / sum(n),
            accuracy = sum(n * accuracy) / sum(n),
            precision = sum(n * precision) / sum(n),
            recall = sum(n * recall) / sum(n),
            f1_score = sum(n * f1_score) / sum(n))

in_df %>% 
  filter(!grepl("8b|llama", tolower(model)),
         prompt == "entailment_binary6") %>% 
  arrange(dataset, model) %>% 
  tidyr::pivot_longer(cols = c(not_parseable, f1_score, precision, recall),
                      names_to = "variable",
                      values_to = "value") %>% 
  ggplot(aes(x = gsub("entailment_", "", variable), y = value, fill = gsub("entailment_", "", model))) +
  geom_bar(position = "dodge", stat = "identity") +
  facet_wrap(~dataset) +
  theme(axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   size = 12),
        strip.text = element_text(size = 15),
        legend.position = "top",
        legend.title = element_blank()) +
  xlab("") +
  ylab("")


in_df %>% 
  filter(!grepl("8b", tolower(model)),
         prompt == "entailment_binary6",
         dataset == "factehr")

in_df %>% 
  tidyr::pivot_longer(cols = c(not_parseable, f1_score, precision, recall),
                      names_to = "variable",
                      values_to = "value") %>% 
  filter(prompt == "entailment_binary6",
         model == 'shc-gpt-4o') %>% 
  ggplot(aes(x = gsub("entailment_", "", variable), y = value, fill = gsub("entailment_", "", prompt))) +
  geom_bar(position = "dodge", stat = "identity") +
  facet_wrap(~dataset + model) +
  theme(axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   size = 12),
        strip.text = element_text(size = 15),
        legend.position = "top",
        legend.title = element_blank()) +
  xlab("") +
  ylab("")

#########
in_path <- "/Users/akshayswaminathan/just-the-facts/data/datasets/output/nli_benchmarking.csv"

df <- readr::read_csv(in_path)

df %>% 
  filter(dataset == "factehr",
         grepl("shc", model)) %>% 
  # filter(not_parseable) %>% 
  sample_n(20) %>% 
  pull(model_output)

######
in_path <- "/Users/akshayswaminathan/just-the-facts/data/datasets/prompted_sampled/entailment_for_now.csv"
other_models_df <- readr::read_csv(in_path)

o1_path <- "~/Downloads/gpt_o1_entailment.csv"
o1_df <- readr::read_csv(o1_path)

in_df <- bind_rows(o1_df, other_models_df) %>% 
  filter(model_name != "final_merged_o1-mini_max8000") %>% 
  select(-`...1`)

in_df %>% 
  group_by(model_name) %>% 
  summarize(n_distinct(uid))

readr::write_csv(in_df, file = "~/Downloads/entailment_output_factehr_10162024.csv")

entailment_prop_df <- in_df %>% 
  mutate(model_name = case_when(grepl("llama", tolower(model_name)) ~ "Llama-3-8b",
                                grepl("gemini", tolower(model_name)) ~ "Gemini-1.5",
                                grepl("shc-gpt-4o", tolower(model_name)) ~ "GPT-4o",
                                grepl("o1-mini", tolower(model_name)) ~ "GPT-o1-mini")) %>% 
  group_by(uid, doc_id, dataset_name, note_type, prompt, entailment_type, model_name) %>% 
  summarise(n = n(),
            entailment_proportion = mean(entailment_pred)) %>% 
  ungroup()

entailment_prop_df %>% 
  group_by(note_type, entailment_type, model_name) %>% 
  summarise(n = sum(n),
            entailment_proportion = mean(entailment_proportion)) %>% 
  rename(value=entailment_proportion) %>%  
  pivot_wider(id_cols = c(note_type, model_name),
              names_from = entailment_type,
              values_from = c(value, n))
  ggplot(aes(x = model_name, y = value, fill = entailment_type)) +
    geom_bar(position = "dodge", stat = "identity") +
    facet_wrap(~note_type) +
    theme(axis.text.x = element_text(angle = 45,
                                     hjust = 1,
                                     size = 12),
          strip.text = element_text(size = 15),
          legend.position = "top",
          legend.title = element_blank()) +
    xlab("") +
    ylab("")
  
  
entailment_by_note <- in_df %>% 
  group_by(uid, model_name, entailment_type, note_type) %>% 
  summarise(n = n(),
            entailment_prop = mean(entailment_pred)) %>%  
  mutate(model_name = case_when(grepl("llama", tolower(model_name)) ~ "Llama-3-8b",
                                grepl("gemini", tolower(model_name)) ~ "Gemini-1.5",
                                grepl("shc-gpt-4o", tolower(model_name)) ~ "GPT-4o",
                                grepl("o1-mini", tolower(model_name)) ~ "GPT-o1-mini"))

entailment_by_note %>%  
  mutate(model_name = case_when(grepl("llama", tolower(model_name)) ~ "Llama-3-8b",
                                grepl("gemini", tolower(model_name)) ~ "Gemini-1.5",
                                grepl("shc-gpt-4o", tolower(model_name)) ~ "GPT-4o",
                                grepl("o1-mini", tolower(model_name)) ~ "GPT-o1-mini")) %>% 
  ggplot(aes(x = model_name, y = entailment_prop, fill = model_name)) +
  geom_boxplot() +
  facet_wrap(~entailment_type + note_type) +
  theme(axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   size = 12),
        strip.text = element_text(size = 15),
        legend.position = "top",
        legend.title = element_blank())

####
in_path <- "/Users/akshayswaminathan/just-the-facts/data/datasets/raw/entailment/factehr/factehr.csv"  
in_df <- readr::read_csv(in_path)  

in_df$human_label %>% mean()

####
in_path <- "/Users/akshayswaminathan/just-the-facts/data/datasets/output/nli_benchmarking.csv"  
in_df <- readr::read_csv(in_path)  
in_df %>% 
  filter(model == "shc-gpt-4o",
         prompt == "entailment_binary6",
         dataset == "factehr") %>% 
  count(label, entailment_pred)

#####
set.seed(1234)

annotations_path <- "~/Downloads/annotation_file_101724.csv"
annotations_df <- readr::read_csv(annotations_path) %>% 
  select(-...1)

annotations_df %>% 
  mutate(model_name = case_when(grepl("llama", tolower(model_name)) ~ "Llama-3-8b",
                                grepl("gemini", tolower(model_name)) ~ "Gemini-1.5",
                                grepl("shc-gpt-4o", tolower(model_name)) ~ "GPT-4o",
                                grepl("o1-mini", tolower(model_name)) ~ "GPT-o1-mini")) %>% 
  select(dataset_name, note_type, entailment_type, model_name) %>% 
  gtsummary::tbl_summary()


base_annotation_df <- annotations_df %>% 
  mutate(unique_id = row_number())

inter_rater_agreement <- base_annotation_df %>% 
  filter(!grepl("medalign", dataset_name)) %>% # filter out medalign because not everyone can annotate that
  sample_n(100)

inter_rater_agreement %>% 
  write.csv("~/Documents/radgraph/annotation_tasks/interrater_agreement_101824.csv")

individual_annotation <- setdiff(base_annotation_df, inter_rater_agreement) %>% 
  mutate(medalign = grepl("medalign", dataset_name))

full_annotation_task <- individual_annotation

full_annotation_task %>% 
  write.csv("~/Documents/radgraph/annotation_tasks/individual_annotation_task_101824.csv")

annotators_initials <- c("AS", "JJ", "IL", "MK", "TO", "LT")
annotators <- tibble(annotator = annotators_initials,
                     annotator_id = seq_along(annotators_initials)) %>% 
  mutate(has_medalign_access = !annotator %in% c("JJ", "LT"))

rows_per_annotator <- (nrow(full_annotation_task) / nrow(annotators)) %>% 
  ceiling()

annotator_assignment <- full_annotation_task %>% 
  arrange(medalign) %>% # assign the non medalign rows first
  mutate(annotator_id = c(rep(annotators$annotator_id[!annotators$has_medalign_access], rows_per_annotator),
                          rep(annotators$annotator_id[annotators$has_medalign_access], rows_per_annotator)) %>% 
           head(nrow(full_annotation_task)))

annotator_counts <- annotator_assignment %>% 
  count(annotator_id, medalign) %>% 
  left_join(annotators)

inter_rater_dfs <- map_dfr(annotators$annotator_id,
                           ~inter_rater_agreement %>% 
                             mutate(annotator_id = .x))

# write the annotation task csvs
task_df <- annotator_assignment %>% 
  bind_rows(inter_rater_dfs) %>% 
  left_join(annotators) %>% 
  transmute(unique_id, annotator, annotator_id, premise, hypothesis, inferrable = NA, notes = NA) %>% 
  arrange(annotator) %>% 
  group_by(annotator) %>% 
  nest() 

map2(task_df$data, task_df$annotator, ~openxlsx::write.xlsx(.x, file = glue::glue("~/Documents/radgraph/annotation_tasks/entailment_annotation_101824_{.y}.xlsx")))


### get new medalign data to annotate
annotation_df <- readr::read_csv("/Users/akshayswaminathan/Downloads/annotation_file_111024.csv")
precision_sentences <- readr::read_csv("/Users/akshayswaminathan/Downloads/release_files_final/precision_hypotheses_110424.csv")
recall_sentences <- readr::read_csv("/Users/akshayswaminathan/Downloads/release_files_final/recall_hypotheses_110424.csv")
notes <- readr::read_csv("/Users/akshayswaminathan/Downloads/release_files_final/combined_notes_110424.csv")
fact_decomps <- readr::read_csv("/Users/akshayswaminathan/Downloads/release_files_final/fact_decompositions_110424.csv") %>% 
  mutate(model_name = case_when(model %in% c("shc-gpt-4o", "gpt-4o") ~ "final_merged_shc-gpt-4o_max4000",
                           model == "meta-llama/Meta-Llama-3-8B-Instruct" ~ "merged_meta-llama_Meta-Llama-3-8B-Instruct_split__max4000",
                           model == "o1-mini" ~ "final_merged_o1-mini_max16000",
                           model == "gemini-1.5-flash-002" ~ "final_merged_gemini-1.5-flash-002_max4000"))


annotation_df_clean <- annotation_df %>% 
  left_join(precision_sentences, c("doc_id", "dataset_name" = "dataset", 
                                   "note_type", "prompt", "index", "entailment_type", "model_name" = "model")) %>% 
  select(-...1, -contains("Unnamed")) %>% 
  left_join(recall_sentences, c("doc_id", "dataset_name" = "dataset", 
                                   "note_type", "prompt", "index", "entailment_type")) %>% 
  select(-...1, -contains("Unnamed")) %>% 
  mutate(hypothesis = coalesce(hypothesis.x, hypothesis.y)) %>% 
  select(-hypothesis.x, -hypothesis.y) %>% 
  left_join(notes, by = c("doc_id", "dataset_name", "note_type")) %>% 
  left_join(fact_decomps, by = c("doc_id", "model_name")) %>% 
  select(-...1, -contains("Unnamed")) %>% 
  distinct()

medalign_annotation_base <- annotation_df_clean %>% 
  filter(dataset_name == "medalign_aaai_pre_v1") %>% 
  mutate(premise = if_else(entailment_type == "precision", note_text, fact_decomp),
         unique_id = row_number()) 

medalign_annotation_base %>% 
  write.csv("~/Documents/radgraph/annotation_tasks/medalign_annotations_101824.csv")
  
medalign_annotation <-  medalign_annotation_base %>% 
  transmute(annotator = "MK", annotator_id = 7, 
            premise, hypothesis, inferrable = NA, notes = NA)

medalign_annotation %>% 
  write.csv("~/Documents/radgraph/annotation_tasks/entailment_annotation_101824_MK.xlsx")




