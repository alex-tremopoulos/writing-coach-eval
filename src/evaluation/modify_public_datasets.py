import pandas as pd

df = pd.read_csv("data/data_routes_original.csv")

df.loc[39, 'target'] = "In distributed systems, data consistency is often difficult to maintain because nodes communicate over an unreliable network, and messages can be delayed or lost. Many algorithms attempt to address this by introducing consensus protocols such as Paxos or Raft, which ensure that all replicas agree on the same value even if some of them fail. However, implementing these protocols is complex and requires careful reasoning about concurrency, since race conditions and deadlocks can easily occur. Furthermore, scaling the system horizontally introduces additional challenges, such as partition tolerance and load balancing, that are not trivial to address in practice."
df.loc[39, 'input'] = "In distributed systems the consistency of data are often difficult to maintain because nodes communicates over unreliable network and messages can delayed or lost. Many algorithm tries to solve this by introducing consensus protocols such as Paxos or Raft, which ensures that all replicas agree on a same value even if some of them fails. However, the implementation of these protocols is complex and requires careful reasoning about concurrency, since race condition and deadlocks can easily occurs. Furthermore, scaling the system horizontally introduce additional challenges, like partition tolerance and load balancing, that is not trivial to address in practice."
df.loc[39, 'query'] = "Fix fluency in this text"
df.loc[39, 'dataset'] = "ChatGPT5.2"
print(df.iloc[39])

df.loc[44, 'query'] ="Are there any other existing methods for leveraging information in knowledge bases for task-specific language model fine-tuning?"
