const admin = require("firebase-admin");
const serviceAccount = require("../../firebase-service-account-key.json"); // Update with your path

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

module.exports = admin;
