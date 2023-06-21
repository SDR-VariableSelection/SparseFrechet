#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

group = [r'$p=3000$', r'$p=4000$', r'$p=6000$']

# files generated from main.py
fname1 = f'n2000_p3000_M1_rho5.txt'
fname2 = f'n2000_p4000_M1_rho5.txt'
fname3 = f'n2000_p6000_M1_rho5.txt'
fplot = f'mulY.pdf'

index = [1,5,9,13,17,21] # general loss
value_vars = ['LASSO',"GLASSO", 
"LLA_E","LLA_G",
"LDA(0)","LDA(G)"]

output = pd.read_csv(fname1, sep=" ", header = None)
out1 = output[index]
out1.columns = value_vars
# Melt the data to long format
long_data1 = pd.melt(out1, value_vars=value_vars,
                    var_name='Method', value_name='Value')
long_data1['Group'] = group[0]
long_data1.head

output = pd.read_csv(fname2, sep=" ", header = None)
out2 = output[index]
out2.columns = value_vars
# Melt the data to long format
long_data2 = pd.melt(out2, value_vars=value_vars,
                    var_name='Method', value_name='Value')
long_data2['Group'] = group[1]
long_data2.head

output = pd.read_csv(fname3, sep=" ", header = None)
out3 = output[index]
out3.columns = value_vars
# Melt the data to long format
long_data3 = pd.melt(out3, value_vars=value_vars,
                    var_name='Method', value_name='Value')
long_data3['Group'] = group[2]
long_data3.head
df_combined = pd.concat([long_data1, long_data2,long_data3], ignore_index=True)
print(df_combined)

# Create side-by-side boxplot
sns.catplot(x="Group", y="Value", hue="Method", kind="box", data=df_combined)
plt.xlabel(r"$n=2000$")
plt.ylabel("General Loss")
# plt.savefig(fplot, format='pdf')
plt.show()

#%%
results = []
index = [1,5,9,13,17,21] # general loss
methods = ['LASSO',"GLASSO", 
"LLA_E","LLA_G", 
"LDA(0)","LDA(G)"]

model = [1,2,3]
for i in np.arange(3):
    for n in np.arange(300, 3000,300):
        fn = 'n'+str(n)+'_p3000_M' + int(model) + '_rho5.txt'
        print(fn)
        output = pd.read_csv(fn, sep=" ", header = None)
        output.iloc[0,1]
        for i in range(output.shape[0]):
            dic = {'id':i,'n': n, 'Example': Ex-10}
            results.append({**dic, 'General Loss': output.iloc[i,index[0]], 'Method': methods[0]})
            results.append({**dic, 'General Loss': output.iloc[i,index[1]], 'Method': methods[1]})
            results.append({**dic, 'General Loss': output.iloc[i,index[2]], 'Method': methods[2]})
            results.append({**dic, 'General Loss': output.iloc[i,index[3]], 'Method': methods[3]})
            results.append({**dic, 'General Loss': output.iloc[i,index[4]], 'Method': methods[4]})
            results.append({**dic, 'General Loss': output.iloc[i,index[5]], 'Method': methods[5]})

df = pd.DataFrame(results)

fig, axs = plt.subplots(1,3, figsize = (20,5))
for i in range(3):
    if i < 2:
        fig = sns.lineplot(data=df[df.Example==i+1], x='n', y='General Loss', #col='Example', 
                    hue='Method', style='Method', markers=True, dashes=False, ax = axs[i], legend=False)
    if i == 2:
        fig = sns.lineplot(data=df[df.Example==i+1], x='n', y='General Loss', #col='Example', 
                    hue='Method', style='Method', markers=True, dashes=False, ax = axs[i])
        plt.setp(fig.get_legend().get_title(), fontsize='15') 
        plt.setp(fig.get_legend().get_texts(), fontsize='10') 
    fig.set_xlabel("n",fontsize=15)
    if i>0:
        fig.set_ylabel("",fontsize=15)
    elif i == 0:
        fig.set_ylabel("General Loss",fontsize=15)
    
    fig.set(title='Example '+str(i+1))
# fig.figure.savefig('relplot_p3000.pdf', bbox_inches='tight')
plt.show()
# %%
