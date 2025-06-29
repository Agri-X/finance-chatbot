module.exports = {
  apps: [
    {
      name: "finance-chatbot",
      script: "uv", 
      args: "run chainlit run main.py",
      exec_mode: "fork", 
      instances: 1,     
    },
  ],
};