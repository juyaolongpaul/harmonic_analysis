\version "2.10.33"

\header {
  title = "59 - Herzliebster Jesu, was hast du"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key bes \major
}


soprano = \relative c'' {
  \partial 4 g4 
    g4 g4 fis4 d4 
  g4 a4 bes4 bes4 
  c4 bes4 a4 a4 
  bes4 c4 d8 c8 bes4 
  ees4 ees4 d4 des8 c8   %5
  c2 bes4 bes4 
  a4 g4 f4 d8 ees8 
  f4 f4 g4 f4 
  ees2 d4 d'4 
  c4 bes8 a8 a2   %10
  g2.
  
}

alto = \relative c' {
  \partial 4 d4 
    ees8 d8 c4 d4 a4 
  d8 e8 fis4 g4 g4 
  a4 g4 fis4 fis4 
  g4 a4 bes4 f4 
  g8 a8 bes4 bes4 bes4   %5
  bes4 a4 f4 g8 f8 
  ees8 d8 e4 d4 a4 
  d8 ees8 f4 f8 ees8 ees8 d8 
  d8 c16 bes16 c4 bes4 bes'4 
  a4 g4 g4 fis4   %10
  d2.
  
}

tenor = \relative c' {
  \partial 4 bes4 
    bes4 a8 g8 a4 fis4 
  g4 c4 d4 d4 
  ees4 d4 d4 d4 
  d4 ees4 f8 ees8 d8 c8 
  bes4 bes8 c8 d8 bes8 g'4   %5
  f4. ees8 d4 d4 
  c4 bes4 a4 f4 
  bes4 bes4 bes4 a4 
  bes4 f4 f4 f'4 
  ees4 d4 e4 d8 c8   %10
  b2.
  
}

baixo = \relative c' {
  \partial 4 g4 
    c,8 d8 ees4 d4 c4 
  bes4 a4 g4 g'4 
  fis4 g4 d4 d4 
  g4 c4 bes4 aes4 
  g4 ges4 f4 e4   %5
  f2 bes,4 g4 
  c4 cis4 d4. c8 
  bes8 c8 d8 bes8 ees4 f4 
  g4 a4 bes4 bes4 
  fis4 g4 cis,4 d4   %10
  g,2. 
  
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