export const defaultLocale = 'zh-cn';
export const defaultHtmlLang = 'zh-CN';

export const locales = ['zh-cn', 'en', 'ja', 'ko', 'fr', 'de', 'es'];

export const translationTargets = [
  { code: 'en', label: 'English' },
  { code: 'ja', label: '日本語' },
  { code: 'ko', label: '한국어' },
  { code: 'fr', label: 'Français' },
  { code: 'de', label: 'Deutsch' },
  { code: 'es', label: 'Español' },
];

export const i18n = {
  defaultLocale,
  locales,
  routing: {
    prefixDefaultLocale: false,
    redirectToDefaultLocale: false,
    fallbackType: 'redirect',
  },
};
