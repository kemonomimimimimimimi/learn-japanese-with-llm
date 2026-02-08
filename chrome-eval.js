const CDP = require('chrome-remote-interface');

(async () => {
  const client = await CDP();
  const {Runtime} = client;
  const expr = process.argv[2];
  const result = await Runtime.evaluate({expression: expr});
  console.log(result.result.value);
  await client.close();
})();
