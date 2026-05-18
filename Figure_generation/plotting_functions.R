## FUNCTIONS FOR PREPARING AND PLOTTING FIGURES

# Generate spearman correlation table with p and Rho values for a list of vars across all TP
get_spearman_table = function(df_list, vars) {
  res_df = data.frame(Timepoint = numeric(), label = character(), var1 = character(), 
                      var2 = character(), p = numeric(), R = numeric())
  for (df_name in names(df_list)) {
    df = df_list[[df_name]]
    for (TP in unique(df$Timepoint)) {
      df_TP = df %>% filter(Timepoint==TP)
      cor_res = stats::cor.test(df_TP %>% pull(vars[1]), df_TP %>% pull(vars[2]),
                                method = 'spearman')
      p = cor_res$p.value
      R=cor_res$estimate
      res_df = rbind(res_df, 
                     data.frame(Timepoint = TP, label = df_name, var1 = vars[1], var2 = vars[2], p = p, R = R))
    }
  }
  return(res_df)
}

# Simple conversion correlation matrix to heatmap
corr_to_heatmap_df = function(input_df, varname, time_label) {
  output_df = input_df %>% data.frame() %>% 
    dplyr::select(!!sym(varname)) %>% 
    dplyr::rename(!!time_label := !!sym(varname)) %>%
    tibble::rownames_to_column("Metabolite") %>% filter(Metabolite != varname)
  return(output_df)
}

# Function for renaming heatmap nutrients
convert_nutrients_heatmap = function(input_df, dict) {
  for (key in names(dict)) {
    input_df = input_df %>% 
      mutate(Metabolite = str_replace(Metabolite, key, dict[[key]]))
  }
  return(input_df %>% tibble::column_to_rownames("Metabolite"))
}

# Function for renaming heatmap proteins
convert_proteins_heatmap = function(input_df, dict) {
  for (key in names(dict)) {
    input_df = input_df %>% 
      mutate(Protein = str_replace(Protein, key, dict[[key]]))
  }
  return(input_df %>% tibble::column_to_rownames("Protein"))
}

# Report wilcox associations
TF_wilcox_message <- function(message, df, idp, dep) {
  wil_res = wilcox.test(df %>% filter(!!sym(idp)==TRUE) %>% pull(!!sym(dep)), df %>% filter(!!sym(idp)==FALSE) %>% pull(!!sym(dep)))
  p = wil_res$p.value
  direction = as.numeric(df %>% filter(!!sym(idp)==TRUE) %>% pull(!!sym(dep)) %>% mean(na.rm = TRUE) < df %>% filter(!!sym(idp)==FALSE) %>% pull(!!sym(dep))  %>% mean(na.rm = TRUE))
  if (p>0.05) {
    return(paste0(message, ": ns"))
  } else {
    return(paste0(message, ": p=", signif(p, 3), " greater in: ", direction))
  }
}

# Calculates negative and positive proteomic correlations for a given metabolite-protein association
get_proteomics_cor_lists = function(Misame_proteome_input, Vital_proteome_input, Child_proteome_input, Feat) {
  res_list = list()
  misame_protein_rcorr_Feat = Misame_proteome_input %>% filter(Timepoint==2) %>%
    dplyr::select(Feat, colnames(misame_proteomics_filt), -SampleID) %>%
    as.matrix() %>% rcorr(type = 'spearman')
  
  misame_Feat_proteomics_assoc <- 
    inner_join(misame_protein_rcorr_Feat$r %>% as.data.frame() %>% dplyr::select(Feat) %>% 
                 tibble::rownames_to_column("ProteinIds"),
               misame_protein_rcorr_Feat$P %>% as.data.frame() %>% 
                 dplyr::rename(Feat = Feat) %>%
                 mutate(p.adjust = p.adjust(Feat, method = "BH")) %>% filter(p.adjust<0.05) %>%
                 dplyr::select(p.adjust) %>% tibble::rownames_to_column("ProteinIds")) %>%
    left_join(misame_protein_meta)
  
  res_list[['misame_pos_assoc']] = misame_Feat_proteomics_assoc %>% filter(!!sym(Feat)>0)
  res_list[['misame_neg_assoc']] = misame_Feat_proteomics_assoc %>% filter(!!sym(Feat)<0)
  
  vital_protein_rcorr_Feat = Vital_proteome_input %>% filter(Timepoint==2) %>%
    dplyr::select(Feat, colnames(vital_proteomics_filt), -SampleID) %>%
    as.matrix() %>% rcorr(type = 'spearman')
  
  vital_Feat_proteomics_assoc <- 
    inner_join(vital_protein_rcorr_Feat$r %>% as.data.frame() %>% dplyr::select(Feat) %>% 
                 tibble::rownames_to_column("ProteinIds"),
               vital_protein_rcorr_Feat$P %>% as.data.frame() %>% 
                 dplyr::rename(Feat = Feat) %>%
                 mutate(p.adjust = p.adjust(Feat, method = "BH")) %>% filter(p.adjust<0.05) %>%
                 dplyr::select(p.adjust) %>% tibble::rownames_to_column("ProteinIds")) %>%
    left_join(vital_protein_meta)
  
  res_list[['vital_pos_assoc']] = vital_Feat_proteomics_assoc %>% filter(!!sym(Feat)>0)
  res_list[['vital_neg_assoc']] = vital_Feat_proteomics_assoc %>% filter(!!sym(Feat)<0)
  
  child_protein_rcorr_Feat = Child_proteome_input %>%
    dplyr::select(Feat, colnames(child_proteomics_filt), -SampleID) %>%
    as.matrix() %>% rcorr(type = 'spearman')
  
  child_Feat_proteomics_assoc <- 
    inner_join(child_protein_rcorr_Feat$r %>% as.data.frame() %>% dplyr::select(Feat) %>% 
                 tibble::rownames_to_column("ProteinIds"),
               child_protein_rcorr_Feat$P %>% as.data.frame() %>% 
                 dplyr::rename(Feat = Feat) %>%
                 mutate(p.adjust = p.adjust(Feat, method = "BH")) %>% filter(p.adjust<0.05) %>%
                 dplyr::select(p.adjust) %>% tibble::rownames_to_column("ProteinIds")) %>%
    left_join(child_protein_meta)
  
  res_list[['child_pos_assoc']] = child_Feat_proteomics_assoc %>% filter(!!sym(Feat)>0)
  res_list[['child_neg_assoc']] = child_Feat_proteomics_assoc %>% filter(!!sym(Feat)<0)
  
  return(res_list)
}

# Function to calculate risk scores
get_risk_scores <- function(input_df, Condition, Exposure, label=Exposure) {
  # Condition must be TRUE for cases (eg. disease) and FALSE for controls 
  # Exposure must be TRUE/FALSE 
  contingency_table = input_df %>% filter(!is.na(!!sym(Exposure))) %>% 
    dplyr::select(SubjectID, !!sym(Exposure), !!sym(Condition)) %>% unique() %>%
    group_by(!!sym(Exposure)) %>% 
    dplyr::summarize(controls=sum(!!sym(Condition)==FALSE), cases=sum(!!sym(Condition)==TRUE)) %>% 
    arrange(!!sym(Exposure)) %>% tibble::column_to_rownames(Exposure) %>% as.matrix() 
  result = riskratio(contingency_table)
  chisq = result$p.value %>% data.frame() %>% dplyr::slice(2) %>% pull("chi.square")
  risk_ratio = result$measure %>% data.frame() %>% dplyr::slice(2) %>% pull("estimate")
  CI_lower  = result$measure %>% data.frame() %>% dplyr::slice(2) %>% pull("lower") 
  CI_upper = result$measure %>% data.frame() %>% dplyr::slice(2) %>% pull("upper") 
  
  chisq_result <- chisq.test(contingency_table, correct = FALSE)
  chisq_stat = unname(chisq_result$statistic)
  chisq_p = unname(chisq_result$p.value)
  
  return(data.frame(Outcome = label, Risk_ratio = risk_ratio, 
                    CI_lower = CI_lower, CI_upper = CI_upper, 
                    Chisq.stat = chisq_stat, Chisq.P=chisq_p))
}

# Prepare metabolite-protein heatmap 
prepare_heatmap_matrix <- function(label, heat_list, Protein_key) {
  
  if (length(unique(str_remove(names(heat_list), ".* ")))>1) {
    # Retrieve Milk and ECM matrices
    milk_matrix <- heat_list[[paste(label, "Milk")]]$r
    ECM_matrix <- heat_list[[paste(label, "ECM")]]$r
    
    milk_matrix_p <- heat_list[[paste(label, "Milk")]]$p
    ECM_matrix_p <- heat_list[[paste(label, "ECM")]]$p
    
    # Bind Milk and ECM matrices, adding empty rows for spacing
    combined_r <- bind_rows(
      add_empty_row(milk_matrix, " "),
      add_empty_row(ECM_matrix, "  "),
      heat_list[[paste(label, "IG")]]$r  # Add IG matrix to the combination
    )
    
    combined_p <- bind_rows(
      add_empty_row(milk_matrix_p, " "),
      add_empty_row(ECM_matrix_p, "  "),
      heat_list[[paste(label, "IG")]]$p  # Add IG statistical matrix
    )
  } else {
    heat_label = names(heat_list)[str_detect(names(heat_list), label)]
    combined_r <- heat_list[[heat_label]]$r
    combined_p <- heat_list[[heat_label]]$p
  }
  
  # Convert using protein key mapping for better readability
  converted_r <- convert_proteins_heatmap(combined_r %>% tibble::rownames_to_column("Protein"), Protein_key)
  converted_p <- convert_proteins_heatmap(combined_p %>% tibble::rownames_to_column("Protein"), Protein_key)
  
  # Rename columns based on MG_heatmap_key for consistency
  colnames(converted_r) <- MG_heatmap_key[colnames(converted_r)]
  colnames(converted_p) <- MG_heatmap_key[colnames(converted_p)]
  
  return(list(r = converted_r, p = converted_p))
}

# Function to calculate the correlations to populate the heatmap
get_heatmap_correlations = function(metab_feats, focus_genes, label, 
                                    include_H1=FALSE, include_cho=FALSE, include_hmo=FALSE) {
  data_map <- list(Misame = list(data = Misame_everything, proteins = misame_proteomics_filt_genes, meta = misame_protein_meta),
                   Vital = list(data = Vital_everything, proteins = vital_proteomics_filt_genes, meta = vital_protein_meta))
  
  if (!label %in% names(data_map)) {
    stop("Unknown label. Please use 'Misame', 'Vital', or 'Child'.")
  }
  
  chosen_data <- data_map[[label]]
  
  col_feats <- focus_genes
  
  if (include_H1) {col_feats = c(col_feats, "H1") } 
  if (include_cho) {col_feats = c(col_feats, "CHO")} # Adds Na:K ratio
  if (include_hmo) {col_feats = c(col_feats, "Total_HMO")} # Adds Na 
  
  focus_genes <- focus_genes[focus_genes %in% colnames(chosen_data$proteins)]
  
  A <- chosen_data$data %>%
    filter(SampleID %in% rownames(chosen_data$proteins)) %>%
    dplyr::select(SampleID, metab_feats, H1, CHO, Total_HMO, Na_K_Ratio, Na) # Adds Na:K ratio and Na
  B <- chosen_data$proteins %>%
    dplyr::select(focus_genes) %>%
    tibble::rownames_to_column("SampleID") %>%
    filter(SampleID %in% chosen_data$data$SampleID) 
  
  AB <- inner_join(B, A) %>%
    mutate_all(~ str_remove_all(., " ")) %>%
    tibble::column_to_rownames("SampleID") %>%
    as.matrix()
  
  cor_matrix <- rcorr(AB, type = "spearman")
  corr_r <- cor_matrix$r %>% as.data.frame() %>% dplyr::select(metab_feats) %>%
    filter(rownames(cor_matrix$r) %in% col_feats)
  corr_p <- cor_matrix$P %>% as.data.frame() %>% dplyr::select(metab_feats) %>%
    filter(rownames(cor_matrix$P) %in% col_feats)
  
  return(list(r = corr_r, p = corr_p))
}

# Protein-vs-protein veresion of get_heatmap_correlations
get_heatmap_correlations_pvp = function(focus_genes, label, MSD_df = data.frame(), MSD_feats,
                                        include_H1=FALSE, include_cho=FALSE, include_hmo=FALSE) {
  
  data_map <- list(Misame = list(data = Misame_everything, proteins = misame_proteomics_filt_genes, meta = misame_protein_meta),
                   Vital = list(data = Vital_everything, proteins = vital_proteomics_filt_genes, meta = vital_protein_meta))
  
  if (!label %in% names(data_map)) {
    stop("Unknown label. Please use 'Misame', 'Vital', or 'Child'.")
  }
  
  chosen_data <- data_map[[label]]
  
  focus_genes <- focus_genes[focus_genes %in% colnames(chosen_data$proteins)]
  col_feats <- focus_genes
  
  if (include_H1) {col_feats = c(col_feats, "H1") } 
  if (include_cho) {col_feats = c(col_feats, "CHO")} # Adds Na:K ratio
  if (include_hmo) {col_feats = c(col_feats, "Total_HMO")} # Adds Na 
  
  A <- chosen_data$data %>%
    filter(SampleID %in% rownames(chosen_data$proteins)) %>%
    dplyr::select(SampleID, H1, CHO, Total_HMO, Na_K_Ratio, Na) # Adds Na:K ratio and Na
  B <- chosen_data$proteins %>%
    dplyr::select(focus_genes) %>%
    tibble::rownames_to_column("SampleID") %>%
    filter(SampleID %in% chosen_data$data$SampleID)
  if (nrow(MSD_df) > 0 ) {
    col_feats = c(col_feats, MSD_feats)
    msd = MSD_df %>% 
      mutate(IgA = ifelse(IgA == "Low Signal", NA, as.numeric(IgA)),
             Calprotectin = ifelse(Calprotectin == "Low Signal", NA, as.numeric(Calprotectin))) %>% 
      dplyr::select(SampleID, MSD_feats)
    A = inner_join(A, msd)
  }
  AB <- inner_join(B, A) %>%
    mutate_all(~ str_remove_all(., " ")) %>%
    tibble::column_to_rownames("SampleID") %>%
    as.matrix()
  
  cor_matrix <- rcorr(AB, type = "spearman")
  corr_r <- cor_matrix$r %>% as.data.frame() %>% dplyr::select(col_feats) %>%
    filter(rownames(cor_matrix$r) %in% col_feats)
  corr_p <- cor_matrix$P %>% as.data.frame() %>% dplyr::select(col_feats) %>%
    filter(rownames(cor_matrix$P) %in% col_feats)
  
  return(list(r = corr_r, p = corr_p))
}

# Function to add an empty row to a dataframe
add_empty_row <- function(df, label) {
  empty_row <- as.data.frame(matrix(0, nrow = 1, ncol = ncol(df)))
  colnames(empty_row) <- colnames(df)
  rownames(empty_row) <- label
  bind_rows(df, empty_row)
}

# Function to assemble complex matrices for heatmap generation
prepare_complete_heatmap_matrix <- function(label, heat_list, Protein_key, proteomics_df, everything_df, MSD_df, complex=FALSE) {
  # Step 1: Prepare Milk and ECM matrices using heat_list
  base_data <- prepare_heatmap_matrix(label, heat_list, Protein_key)
  milk_and_ecm_r <- base_data$r
  milk_and_ecm_p <- base_data$p
  
  # Step 2: Run rcorr for the custom section
  rcorr_output <- everything_df %>%
    filter(SampleID %in% rownames(proteomics_df)) %>%
    left_join(MSD_df) %>% 
    mutate(
      IgA = ifelse(IgA == "Low Signal", NA, as.numeric(IgA)),
      Calprotectin = ifelse(Calprotectin == "Low Signal", NA, as.numeric(Calprotectin))
    ) %>%
    dplyr::select(SampleID, metab_feats, IgA, Calprotectin) %>%
    mutate_all(~ str_remove_all(., " ")) %>%
    tibble::column_to_rownames("SampleID") %>%
    as.matrix() %>%
    rcorr(type = "spearman")
  
  # Extract r and p matrices
  cor_r <- rcorr_output$r %>%
    as.data.frame() %>%
    dplyr::select(metab_feats) %>%
    filter(rownames(rcorr_output$r) %in% c("IgA", "Calprotectin"))
  
  cor_p <- rcorr_output$P %>%
    as.data.frame() %>%
    dplyr::select(metab_feats) %>%
    filter(rownames(rcorr_output$P) %in% c("IgA", "Calprotectin"))
  
  # Append IgG mean correlations dynamically inside function
  if (complex == TRUE) {
    # Dynamically calculate IGG Mean correlations
    igg_mean_r <- data.frame(t(colMeans(cor_r[sapply(cor_r, is.numeric)])))
    rownames(igg_mean_r) <- "IgG (Mean)"
    
    igg_mean_p <- data.frame(t(data.frame(
      "IgG (Mean)" = as.matrix(ifelse(colMeans(cor_p < 0.05) > 0.5, -99, -NA)),
      check.names = FALSE
    ))) %>%
      mutate(across(everything(), as.numeric))
    
    # Combine IgA, Calprotectin, and the IGG Mean row for both correlation and p-value matrices
    custom_r <- rbind(cor_r, igg_mean_r)
    custom_p <- rbind(cor_p, igg_mean_p)
    
    # Assign column names using the heatmap key
    colnames(custom_r) <- MG_heatmap_key[colnames(custom_r)]
    colnames(custom_p) <- MG_heatmap_key[colnames(custom_p)]
    
    # Combine Milk+ECM matrices with custom entries
    final_r <- rbind(milk_and_ecm_r, custom_r)
    final_p <- rbind(milk_and_ecm_p, custom_p)
  } else {
    # Default output when `complex == FALSE`
    final_r <- milk_and_ecm_r
    final_p <- milk_and_ecm_p
  }
  
  # Map protein names using the provided key
  final_r <- convert_proteins_heatmap(final_r %>% tibble::rownames_to_column("Protein"), Protein_key)
  final_p <- convert_proteins_heatmap(final_p %>% tibble::rownames_to_column("Protein"), Protein_key)
  
  return(list(r = final_r, p = final_p))
}

# Function to render heatmap with color and statistical annotations
render_heatmap <- function(matrix_r, matrix_p, title, colors) {
  Heatmap(matrix_r,
          cell_fun = function(j, i, x, y, width, height, fill) {
            row_name <- rownames(matrix_r)[i]
            if (row_name %in% c(" ", "  ", "   ")) {
              return()
            }
            if (!is.na(matrix_p[i, j]) && matrix_p[i, j] == -99) {
              grid.text(sprintf("%.2f", matrix_r[i, j]), x, y, gp = gpar(fontsize = 8))
            } else if (!is.na(matrix_p[i, j]) && matrix_p[i, j] < 0.05) {
              grid.text(if (matrix_r[i, j] == 0) "" else sprintf("%.2f", matrix_r[i, j]), x, y, gp = gpar(fontsize = 8))
            } else {
              grid.text("ns", x, y, gp = gpar(fontsize = 8))
            }
          },
          column_title = title,
          name = "Rho",
          cluster_columns = FALSE,
          cluster_rows = FALSE,
          column_names_rot = -45,
          col = colors
  )
}

# Function to convert p-values to astrices
pval_to_stars <- function(p) {
  sapply(p, function(x) {
    if (is.na(x)) return(NA)
    if (x < 0.001) return("***")
    if (x < 0.01) return("**")
    if (x < 0.05) return("*")
    if (x < 0.1) return(".")
    return("")
  })
}