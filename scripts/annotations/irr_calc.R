library(dplyr)
library(purrr)
library(irr)

# annotator agreement
annotations_path <- "~/Documents/radgraph/annotation_task_completed_11_2024/"
annotation_exercise_path <- "~/Documents/radgraph/annotation_tasks/interrater_agreement_101824.csv"
annotator_file_names <- paste0(annotations_path, list.files(annotations_path))
full_annotation_exercise <- readr::read_csv(annotation_exercise_path)

mode_ <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
calculate_fleiss_kappa <- function(df) {
  # df should contain columns "row_number", "annotator", and "inferrable"
  
  ratings <- df %>% 
    pivot_wider(id_cols = c(unique_id),
                names_from = annotator,
                values_from = inferrable) %>% 
    select(-unique_id) %>% 
    as.matrix()
  
  fleiss_kappa <- kappam.fleiss(ratings, exact = FALSE, detail = FALSE) %>% 
    map(~.x) %>% 
    data.frame()
  
  return(fleiss_kappa$value)
  
}


annotator_dfs <- map(annotator_file_names, readxl::read_xlsx) %>% 
  setNames(list.files(annotations_path))

cleaned_df <- annotator_dfs %>% 
  imap_dfr(~.x %>% 
             mutate(annotator = .y,
                    inferrable = coalesce(as.numeric(inferrable), 0))) 
  # inner_join(full_annotation_exercise, by = c("unique_id"))

### Write all annotations to the final DF
final_human_annotations <- cleaned_df %>% 
  filter(!annotator_id %in% c(7)) %>%  # Mehr's medalign annotations
  group_by(unique_id) %>% 
  summarise(human_pred = mode_(inferrable))

write.csv(final_human_annotations, "~/Documents/radgraph/annotation_tasks/human_annotations_merged_120824.csv")
###

kappa_df <- cleaned_df %>% 
  filter(!annotator_id %in% c(4, 7)) %>% # these are Mehr's IDs which were excluded from latest round of IRR analysis
  group_by(unique_id) %>% 
  filter(n() > 1) %>% 
  ungroup()

kappa_df %>% 
  select(unique_id, annotator_id, premise, hypothesis, inferrable, annotator) %>% 
  write.csv("~/Downloads/factehr_irr_annotations_120824.csv")

kappa_df %>% 
  calculate_fleiss_kappa()

### OLD ###
# group_by_vars <- c("dataset", "prompt")
# 
# map(group_by_vars,
#     ~kappa_df %>% 
#       group_by(!!sym(.x)) %>% 
#       nest() %>% 
#       mutate(kappa = map_dbl(data, calculate_fleiss_kappa),
#              n = map_dbl(data, nrow)) %>% 
#       select(-data))
# 
# kappa_df %>% 
#   filter(prompt == "precision") %>%
#   pivot_wider(id_cols = c(row_number),
#               names_from = annotator,
#               values_from = inferrable) %>% 
#   select(-row_number) %>% 
#   as.matrix() %>% 
#   kappam.fleiss()
# 
# 
# test = kappa_df %>% 
#   filter(prompt == "precision") %>%
#   pivot_wider(id_cols = c(row_number),
#               names_from = annotator,
#               values_from = inferrable) %>% 
#   select(-row_number) %>% 
#   as.matrix() 
# 
# mean(test)
# 
# test[1,4] = 0
# kappam.fleiss(test)
  
    