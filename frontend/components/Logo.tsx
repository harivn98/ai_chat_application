/** The ACAP mark: a chat bubble holding lines of document text, with a spark for the AI. Same drawing as app/icon.svg. */
export default function Logo({ size = 26 }: { size?: number }) {
  return (
    <svg className="logo" width={size} height={size} viewBox="0 0 32 32" aria-hidden>
      <defs>
        <linearGradient id="acap-logo-bg" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#3d86f5" />
          <stop offset="1" stopColor="#1747a6" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="9" fill="url(#acap-logo-bg)" />
      <path d="M10 8h10a3 3 0 0 1 3 3v7a3 3 0 0 1-3 3h-6l-4 3.5V21a3 3 0 0 1-3-3v-7a3 3 0 0 1 3-3z" fill="#fff" />
      <path d="M11 12.5h8M11 16.5h5" stroke="#1f5fd6" strokeWidth="1.8" strokeLinecap="round" />
      <path
        d="M24.5 3.8c.35 2.3.9 2.85 3.2 3.2-2.3.35-2.85.9-3.2 3.2-.35-2.3-.9-2.85-3.2-3.2 2.3-.35 2.85-.9 3.2-3.2z"
        fill="#ffd166"
        stroke="#1f5fd6"
        strokeWidth="1"
        strokeLinejoin="round"
        paintOrder="stroke"
      />
    </svg>
  );
}
