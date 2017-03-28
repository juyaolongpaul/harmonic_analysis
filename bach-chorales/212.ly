\version "2.10.33"

\header {
  title = "212 - Herr, ich denk' an jene Zeit"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key ees \major
}


soprano = \relative c' {
  ees4 ees4 bes'4 bes4 
  c4 d4 ees2 
  d4. ees8 f4 bes,4 
  ees4 d4 c2 
  bes2 ees4 ees4   %5
  bes4 bes4 aes4 aes4 
  g2 bes4 bes4 
  aes4 g4 f2 
  ees2 f4 f4 
  g8 a8 bes4 bes4 a4   %10
  bes2 c4 d4 
  ees2 bes4 bes4 
  aes4 g4 f2 
  ees1 
  
}

alto = \relative c' {
  bes4 bes4 ees4 g8 f8 
  ees4 f4 g2 
  f4 bes4. aes8 g4~ 
  g8 f8 f4 g4 f8 ees8 
  d2 g4 g4   %5
  f4 f8 g4 c,8 f8 ees16 d16 
  ees2 f4 ees4 
  f4. ees8 ees4 d4 
  bes2 d4 d4 
  ees4 f8 g8 f4 f4   %10
  f2 f8 aes4 g8 
  g2 f8 aes4 g8 
  f4. ees4 d16 c16 d4 
  bes1 
  
}

tenor = \relative c' {
  g4 g8 aes8 bes8 ees4 des8 
  c8 bes8 aes4 bes2 
  bes4 bes4 bes4 bes4~ 
  bes8 a8 bes2 a4 
  f2 bes4 bes8 c8   %5
  d4. c8 c4 bes4 
  bes2 d8 c8 bes4 
  bes4 bes4 c8 f,8 bes8 aes8 
  g2 bes4. aes8 
  bes8 c8 d4 c8 d8 ees4   %10
  d2 c8 f8 d4 
  c2 f4. ees8~ 
  ees8 d16 c16 bes4 c4 bes8 aes8 
  g1 
  
}

baixo = \relative c {
  ees4. f8 g4 ees4 
  aes8 g8 f4 ees2 
  bes'8 aes8 g4 d4 ees8 d8 
  c8 f8 bes,8 d8 ees8 c8 f4 
  bes,2 ees,8 f8 g8 aes8   %5
  bes8 c8 d8 ees8 f8 ees8 d8 bes8 
  ees2 bes'8 aes8 g8 f8 
  ees8 d8 ees4 aes,4 bes4 
  ees,2 bes''8 aes8 g8 f8 
  ees4 d8 ees8 f2   %10
  bes,2 aes'4 b,4 
  c2 d4 ees4 
  f4 g4 aes4 bes8 bes,8 
  ees1 
  
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