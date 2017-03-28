\version "2.10.33"

\header {
  title = "8 - Freuet euch, ihr Christen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \minor
}


soprano = \relative c'' {
  c4 c4 bes4 aes4 
  g4. f8 f4 f4 
  aes4. bes8 c4 c4 
  bes4. aes8 aes2 
  aes4 bes4 bes4 aes4   %5
  g4. f8 f2 
  c'4 c4 des4 des4 
  bes4 bes4 c4 c4 
  aes4 aes4 des4 des4 
  c4 c4 bes2   %10
  g4 aes4 bes4 aes4 
  g4 g4 f2 
  f'4 f4 ees4 ees4 
  d4 d4 c4 c4 
  c4 c4 des4 c4   %15
  bes4. aes8 aes4 aes4 
  aes4 aes4 des4 des4 
  bes4 bes4 ees4 ees4 
  c4 c4 f4 f4 
  e4 e4 f4 f4   %20
  
}

alto = \relative c' {
  f4 f4 f8 e8 f4 
  f8 g8 e4 f4 c4 
  f4. g8 aes4 aes4 
  aes8 f8 g4 ees2 
  f4 g4 e4 f4   %5
  f4 e4 c2 
  f4 f4 f4 f4 
  g8 aes8 bes8 g8 aes4 aes4 
  f4 f4 bes4 bes4 
  bes4 a4 f2   %10
  e4 f4 f8 e8 f4 
  f4 e4 c2 
  aes'4 aes4 g4 g4 
  g8 aes8 g8 f8 e4 e4 
  f4 f4 f8 g8 aes4   %15
  aes8 f8 g4 ees4 ees4 
  f4 f4 bes4 bes4 
  g4 g4 c4 c4 
  aes8 g8 aes8 bes8 c4 bes8 aes8 
  g4 g4 a4 a4   %20
  
}

tenor = \relative c' {
  aes4 a4 bes4 c4 
  des4 c8 bes8 aes4 aes4 
  c4. bes8 ees4 ees4 
  f4 ees8 des8 c2 
  des4 des4 g,4 f8 aes8   %5
  des4 c8 bes8 aes2 
  a8 bes8 c8 a8 bes4 bes4 
  bes4 ees4 ees4 ees4 
  des4 des4 f4 f4 
  ges4 f8 ees8 des2   %10
  c4 c4 bes4 c4 
  des4 c8 bes8 aes2 
  c4 c4 c4 c4 
  c4 b4 c4 g4 
  a4 a4 bes4 ees,4   %15
  ees'4. des8 c4 c4 
  des4 des4 f4 f4 
  ees4 ees4 g4 g4 
  f4 f4 c4 f,4 
  c'4 c4 c4 c4   %20
  
}

baixo = \relative c {
  f4 ees4 des4 c4 
  bes4 c4 f,4 f4 
  f'4 ees8 des8 c8 bes8 aes4 
  des4 ees4 aes,2 
  des4 c8 bes8 c4 des4   %5
  bes4 c4 f,2 
  f'8 g8 a8 f8 bes8 c8 des8 bes8 
  ees,8 f8 g8 ees8 aes4 aes4 
  des,8 ees8 f8 des8 bes8 c8 des8 bes8 
  ees8 c8 f4 bes,2   %10
  bes'4 aes4 g4 f4 
  bes,4 c4 f,2 
  f8 g8 aes8 bes8 c8 d8 ees8 f8 
  g4 g,4 c4 c4 
  f8 ees8 des8 c8 bes4 c8 des8   %15
  ees4 ees4 aes,4 aes4 
  des8 ees8 des8 c8 bes8 c8 des8 bes8 
  ees8 f8 ees8 des8 c8 des8 ees8 c8 
  f8 e8 f8 g8 aes8 g8 aes8 bes8 
  c4 c,4 f4 f4   %20
  
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