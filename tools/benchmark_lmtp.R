# Fit lmtp's TMLE to the wide panel written by tools/benchmark_lmtp.py.
# Usage: Rscript tools/benchmark_lmtp.R <prefix> <tau> [folds]
suppressMessages({library(lmtp); library(SuperLearner)})
args <- commandArgs(TRUE)
prefix <- args[1]; tau <- as.integer(args[2])
folds <- if (length(args) >= 3) as.integer(args[3]) else 2L
d <- read.csv(paste0(prefix, "_wide.csv"))
baseline <- readLines(paste0(prefix, "_baseline.txt"))
lev <- c("none", "transfer", "escalation", "requeue", "shift_change")
A <- paste0("A_", 1:tau)
for (a in A) d[[a]] <- factor(d[[a]], levels = lev)
L <- c("queue_length", "log_wait", "shift_time_left", "skill_mismatch",
       "with_specialist", "sentiment", "log_idle", "cum_transfer",
       "cum_escalation", "cum_requeue", "cum_shift_change")
time_vary <- lapply(1:tau, function(t) paste0(L, "_", t))

never <- function(levels) {
  function(data, trt) {
    x <- data[[trt]]
    x[!is.na(x) & x %in% levels] <- "none"
    factor(x, levels = lev)
  }
}
policies <- list(
  natural_course = NULL,
  never_escalation = never("escalation"),
  never_requeue = never("requeue"),
  never_any = never(lev[-1])
)
fit <- function(shift) {
  lmtp_tmle(d, trt = A, outcome = paste0("Y_", 1:tau), baseline = baseline,
            time_vary = time_vary, cens = paste0("C_", 1:tau),
            compete = paste0("D_", 1:tau), shift = shift, mtp = TRUE, k = 1,
            outcome_type = "survival", folds = folds,
            learners_outcome = "SL.glm", learners_trt = "SL.glm")
}
fits <- list()
for (nm in names(policies)) {
  t0 <- Sys.time()
  fits[[nm]] <- fit(policies[[nm]])
  cat(nm, format(Sys.time() - t0), "\n")
}
rows <- lapply(names(fits), function(nm) {
  f <- fits[[nm]]
  est <- tidy(f)
  rd_se <- if (nm == "natural_course") 0 else {
    ct <- lmtp_contrast(f, ref = fits$natural_course)
    as.numeric(ct$estimates$std.error)
  }
  # lmtp reports survival; the risk of a complaint is 1 - survival.
  data.frame(regime = nm, risk = 1 - est$estimate, se = est$std.error, rd_se = rd_se)
})
write.csv(do.call(rbind, rows), paste0(prefix, "_lmtp.csv"), row.names = FALSE)
