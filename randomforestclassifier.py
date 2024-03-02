from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier()       
rfc.fit(X_train, y_train)       
y_true = y_test       
y_pred = rfc.predict(X_test)       
rfc_tn, rfc_fp, rfc_fn, rfc_tp = confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel()