import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 5,
  duration: '30s',
  thresholds: {
    http_req_failed: ['rate<0.05'],
    http_req_duration: ['p(95)<1000']
  }
};

const BASES = [
  { name: 'ingestion', url: 'http://localhost:8001' },
  { name: 'agent', url: 'http://localhost:8005' },
  { name: 'nlu', url: 'http://localhost:8003' },
  { name: 'rag', url: 'http://localhost:8004' },
];

export default function () {
  // Hit health endpoints
  for (const s of BASES) {
    const res = http.get(`${s.url}/health`, { headers: { 'X-Request-ID': `${__VU}-${__ITER}` } });
    check(res, {
      [`${s.name} status 200/healthy`]: (r) => r.status === 200 && r.body.includes('status'),
    });
  }

  // Light metrics scrape
  for (const s of BASES) {
    const res = http.get(`${s.url}/metrics`);
    check(res, { [`${s.name} metrics available`]: (r) => r.status === 200 });
  }

  sleep(1);
}
