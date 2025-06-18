from matplotlib import pyplot as plt

weights = model._edge_weights

plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16

plt.figure(figsize=(4, 2.5), dpi=768)
plt.hist(weights, 30)
plt.yscale('log')
plt.xlabel('Weight', fontsize=16)
plt.ylabel('#Token transfer', fontsize=16)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig('log_weight.pdf')

unlog_ws = [10 ** w for w in weights]
plt.figure(figsize=(4, 2.5), dpi=768)
plt.hist(unlog_ws, 30, color='gray')
plt.yscale('log')
plt.xlabel('Value', fontsize=16)
plt.ylabel('#Token transfer', fontsize=16)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig('unlog_weight.pdf')
