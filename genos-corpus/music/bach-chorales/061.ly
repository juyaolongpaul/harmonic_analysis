\version "2.10.33"

\header {
  title = "61 - Jesu Leiden, Pein un Tod"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key ees \major
}


soprano = \relative c'' {
  g4 f4 ees4 f4 
  g4 aes4 bes2 
  c4 bes4 aes4 g8 f8 
  f2 ees2 
  g4 bes4 aes4 g4   %5
  f4 ees4 d2 
  ees4 f4 g4 g4 
  aes2 g2 
  g4 aes4 bes4 aes4 
  g4 f4 g2   %10
  f4 f4 g8 a8 bes4 
  bes4 a4 bes2 
  bes4 g4 c4 bes4 
  aes4 g4 f2 
  g8 aes8 bes4 aes4 g8 f8   %15
  f2 ees2 
  
}

alto = \relative c' {
  ees4 d4. c4 bes8 
  bes4 aes4 d2 
  ees8 aes4 g4 f8 ees4~ 
  ees4 d4 bes2 
  ees4 f4 ees8 f8 g8 ees8   %5
  d4 c4 b2 
  c4 d4 ees4 ees4~ 
  ees8 d16 ees16 f8 d8 ees2 
  ees4 ees4 e4 f4 
  des8 c8 c8 d8 ees2   %10
  c4 c8 d8 ees4 des8 ees8 
  f2 f2 
  f4 ees8 f8 g4 g4 
  ees8 f8 bes,8 c8 d2 
  ees4. bes8 ees8 f8 g8 c,8   %15
  d8 ees16 c16 d4 bes2 
  
}

tenor = \relative c' {
  bes4 bes8 aes8 g4 f4 
  ees8 ees'4 des8 ees2 
  c4 d4 ees8 d8 ees8 c8 
  bes4. aes8 g2 
  bes4 bes4 c8 d8 ees8 c8   %5
  aes8 g8 g4 g2 
  g4 bes4 bes4 bes4 
  c8 aes8 f8 bes8 bes2 
  c4 c4 bes4 c4 
  bes4 aes4 g2   %10
  aes8 bes8 c8 bes8 bes4. c8 
  des4 c4 d2 
  bes4 bes4 ees4 d8 bes8 
  c8 d8 ees4 bes2 
  bes8 aes8 g8 f8 ees8 d8 ees4   %15
  bes'4. aes8 g2 
  
}

baixo = \relative c {
  ees4 bes4 c4 d4 
  ees4 f4 g2 
  aes4 bes4 c4 bes8 aes8 
  bes4 bes,4 ees2 
  ees8 d8 d8 c8 c2~   %5
  c8 b8 c8 ees8 g2 
  c,8 c'8 bes8 aes8 g8 f8 ees8 d8 
  c8 f8 d8 bes8 ees2 
  c4 bes8 aes8 g8 g'4 f8~ 
  f8 e8 f4 c2   %10
  f8 g8 aes4 g4 ges4 
  f8 ees8 f4 bes,2 
  d8 bes8 ees4. c8 g'4~ 
  g8 f8 g8 aes8 bes2 
  ees,4. d8 c8 c'8 bes8 aes8   %15
  bes4 bes,4 ees2 
  
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