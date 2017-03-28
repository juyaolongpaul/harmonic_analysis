\version "2.10.33"

\header {
  title = "244 - Jesu, Jesu, du bist mein"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  g4. c8 b4 c4 
  d4 ees8 d8 d2 
  ees4 c4 f4 ees4 
  d4 d8. ees16 c4 c4 
  g4 g4 c4 bes4   %5
  aes4 g8 f8 f2 
  bes4 g4 c8 d8 ees4 
  g,4 f8 ees8 ees4 ees4 
  g4. a8 b4 b4 
  c4 d4 b4 b4   %10
  c4 d4 ees4 c4 
  f4 ees4 d4 d4 
  c4 g4 aes4 g4 
  g4. f8 g2 
  c4 g4 aes8 g8 f8 ees16 d16   %15
  ees4 d8 c8 c2 
  
}

alto = \relative c' {
  ees4 g4 f4 g4 
  g4 g4 g2 
  g4 g4 f8 g8 aes4~ 
  aes8 d,8 g8 f8 ees4 ees4 
  ees8 f8 g4 c,8 d8 ees4   %5
  f4 ees4 d2 
  f4. ees8 ees8 f8 g8 f8 
  ees4 d4 bes4 bes4 
  ees4 ees4 f4 g4 
  g4 aes4 g4 g4   %10
  g4 g4 g4 g4 
  f4 g4 g4 g4 
  g4 g4. f4 ees8 
  ees8 d8 c4 d2 
  g8 f8 g8 ees8 f4 c4   %15
  c4 b4 g2 
  
}

tenor = \relative c' {
  c4 c4 d4 ees4 
  d4 c4 c4 b4 
  c4 ees8 d8 c4 c4 
  c4 b4 g4 g4 
  c4 d4 ees4 bes4   %5
  bes4 bes4 bes2 
  bes4 bes4 c8 bes4 c8 
  bes4 bes8. aes16 g4 g4 
  bes4 c8 ees8 d4 d4 
  ees4 f4 d4 d4   %10
  ees4 d4 c8 d8 ees8 d8 
  c4 c4 c4 b4 
  c4 ees4 c4 b4 
  c8 b8 c4 b2 
  c4 c4 c4 aes4   %15
  g4 g8. f16 e2 
  
}

baixo = \relative c {
  c4 ees4 d4 c4 
  b4 c4 g2 
  c4 c'8 bes8 aes4. g8 
  f4 g4 c,2 
  c'4 bes4 aes4 g8 f8   %5
  ees8 d8 ees4 bes2 
  d8 bes8 ees4 aes,4 g8 aes8 
  bes8 aes8 bes4 ees,2 
  ees'8 d8 ees8 c8 d8 aes'8 g8 f8 
  ees8 c8 f,8 f'8 g8 g,8 g'8 f8   %10
  ees8 d8 c8 b8 c8 g'8 c8 bes8 
  aes4 g8 f8 g4 g,4 
  ees'4 c4 f4 g4 
  aes2 g2 
  ees8 d8 ees8 c8 f8 g8 aes8 f8   %15
  g4 g,4 c2 
  
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