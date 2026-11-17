export const defaultLocale = 'zh-cn';
export const defaultHtmlLang = 'zh-CN';

export const locales = ['zh-cn', 'en'];

export const localeLabels = {
  'zh-cn': '中文',
  en: 'English',
};

export const localeHtmlLangs = {
  'zh-cn': 'zh-CN',
  en: 'en',
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
