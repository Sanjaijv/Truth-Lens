FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Ensure the upload folder exists
RUN mkdir -p uploads

EXPOSE 5001

CMD ["node", "src/server.js"]
