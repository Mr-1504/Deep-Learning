start = time.time()

m = Prophet.Prophet(changepoint_prior_scale=0.01)
m.fit(train_df)

end = time.time()
print('Time Required :', end-start,'seconds')

test = test_df.iloc[:,0:1]
fcst = m.predict(test)

fig = m.plot(fcst)