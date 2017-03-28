\version "2.10.33"

\header {
  title = "113 - Christus, der uns selig macht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \minor
}


soprano = \relative c'' {
  f4 f4 f4 f4 
  ees4 des4 c2 
  des4 ees4 f4 f4 
  ees4 des4 c2 
  bes4 c4 des4 bes4   %5
  bes8 aes8 ges4 f2 
  ges4 aes4 bes4 bes4 
  aes4 ges4 f2 
  bes4 a4 bes4 c4 
  des4 c8 bes8 bes2   %10
  des4 ees4 f4 f4 
  ees4 des8 c8 c2 
  bes4 c4 des8 c8 bes4 
  bes8 aes8 ges4 f2 
  ges4 aes4 bes4 bes4   %15
  aes4 ges4 f2~ 
  f1 
  
}

alto = \relative c'' {
  a4 a4 bes4 c4 
  bes4 bes4 a2 
  f4 ges4 aes4 aes4 
  ges4 f4 f2 
  f4 f4 f4 ges4   %5
  f4 ees4 d2 
  ees4 ees4 e4 e4 
  ees2 d2 
  f4 ees4 f4 ges4 
  f4 f4 ges2   %10
  aes4 aes4 aes4 bes4 
  bes2 a2 
  f4 f4 f4 ges4 
  f4 ees4 d2 
  ees4 f4 ges4 ges4   %15
  f4 ees2 des4 
  c1 
  
}

tenor = \relative c' {
  c4 c4 des4 c8 des8 
  ees8 f8 ges4 c,2 
  bes4 bes4 ces4 bes4 
  bes2 a2 
  des4 c4 bes8 c8 des8 ees8   %5
  f8 bes,8 bes4 bes2 
  bes4 ces4 des4 des4 
  ces8 des8 ees4 bes2 
  des4 c4 bes4 bes4 
  bes8 aes8 des4 des2   %10
  f4 c4 des4 des4 
  ees8 f8 ges4 f2 
  bes,4 a4 bes8 c8 des8 ees8 
  f8 bes,8 bes4 bes2 
  bes4 des4 des4 des4   %15
  des8 ces8 bes4 a4 bes4 
  a1 
  
}

baixo = \relative c {
  f4 f4 bes4 aes4 
  ges4 f8 ees8 f2 
  bes4 ges4 d4 d4 
  ees4 bes4 f'2 
  bes4 a4 bes4 ges4   %5
  d4 ees4 bes2 
  ees4 ces4 g4 g4 
  aes4 a4 bes2 
  bes4 c4 des4 ees4 
  f4 des4 ges2   %10
  f4 aes4 des8 c8 bes8 aes8 
  ges4 f8 ees8 f2 
  des4 f4 bes8 aes8 ges4 
  d4 ees4 bes2 
  ees4 des4 ges,8 aes8 bes8 ces8   %15
  des4 ees4 f2~ 
  f1 
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}