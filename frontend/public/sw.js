const CACHE_NAME = 'sign2speech-v1';
const ASSETS = [
  '/',
  '/index.html',
  '/logo.jpg',
  '/male_signing.jpg',
  '/favicon.svg'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
});

self.addEventListener('fetch', (event) => {
  // Only intercept GET requests for static assets, leave APIs and real-time streams to network
  if (event.request.method !== 'GET' || event.request.url.includes('/api/')) {
    return;
  }
  event.respondWith(
    caches.match(event.request).then((response) => response || fetch(event.request))
  );
});
