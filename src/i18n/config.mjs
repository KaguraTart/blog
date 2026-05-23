export const defaultLocale = 'zh-cn';
export const defaultHtmlLang = 'zh-CN';

export const locales = ['zh-cn', 'en', 'ja', 'ko', 'fr', 'de', 'es'];

export const localeLabels = {
  'zh-cn': '中文',
  en: 'English',
  ja: '日本語',
  ko: '한국어',
  fr: 'Français',
  de: 'Deutsch',
  es: 'Español',
};

export const localeHtmlLangs = {
  'zh-cn': 'zh-CN',
  en: 'en',
  ja: 'ja',
  ko: 'ko',
  fr: 'fr',
  de: 'de',
  es: 'es',
};

export const routeLocales = locales.filter(locale => locale !== defaultLocale);

export const siteLocales = locales.map(code => ({
  code,
  label: localeLabels[code] ?? code,
  htmlLang: localeHtmlLangs[code] ?? code,
}));

export const i18n = {
  defaultLocale,
  locales,
  routing: {
    prefixDefaultLocale: false,
    redirectToDefaultLocale: false,
    fallbackType: 'redirect',
  },
};
