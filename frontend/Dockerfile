FROM node:18-alpine

WORKDIR /app

# Copy package and install
COPY package.json package-lock.json ./
RUN npm install

# Copy source
COPY . .

EXPOSE 5173

# Start dev server
CMD ["npm", "run", "dev", "--", "--host"]
